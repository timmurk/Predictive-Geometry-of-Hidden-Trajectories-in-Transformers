import json
import math
import os

import torch

from pgeom.common import base_parser, setup, examples, save_rows, save_json
from pgeom.data import load_sequences
from pgeom.geometry import PullbackFisher, top_eig, hutchinson, rank_for_tau
from pgeom.lowrank import compress, match_budget, target_linears
from pgeom.model import Trace, load_model, get_layers


def spectrum(args, model, tok):
    n_layers = len(get_layers(model))
    vals = [[] for _ in range(n_layers)]
    traces = [[] for _ in range(n_layers)]
    for i, ids, t in examples(args, tok):
        tr = Trace(model, ids[:, : t + 1])
        for layer in range(n_layers):
            op = PullbackFisher(tr.downstream(layer, [t]), tr.hidden(layer))
            v, _ = top_eig(op, args.kmax, args.iters)
            vals[layer].append(v.cpu())
            traces[layer].append(hutchinson(op, args.probes)[0])
            del op
        print(args.name, "spectrum example", i, "done")
    return dict(
        eigs=[torch.stack(v).mean(0).tolist() for v in vals],
        trace=[sum(t) / len(t) for t in traces],
    )


@torch.no_grad()
def perplexity(model, seqs, device):
    nll, count = 0.0, 0
    for ids in seqs:
        ids = ids[None].to(device)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        lp = torch.log_softmax(logits[0, :-1], -1)
        nll -= lp.gather(-1, ids[0, 1:, None]).sum().item()
        count += ids.shape[1] - 1
    return math.exp(nll / count)


def finetune(model, seqs, steps, lr, ratio, device):
    params_a, params_b = [], []
    for name, prm in model.named_parameters():
        if name.endswith(".a"):
            params_a.append(prm)
        elif name.endswith(".b"):
            params_b.append(prm)
    for prm in params_a + params_b:
        prm.requires_grad_(True)
    opt = torch.optim.AdamW([
        {"params": params_a, "lr": lr * ratio},
        {"params": params_b, "lr": lr},
    ], weight_decay=0.0)
    model.train()
    for step in range(steps):
        ids = seqs[step % len(seqs)][None].to(device)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()
    for prm in params_a + params_b:
        prm.requires_grad_(False)


def full_rank(model):
    layer = get_layers(model)[0]
    return min(min(m.weight.shape) for _, m in target_linears(layer))


def main():
    p = base_parser()
    p.add_argument("--kmax", type=int, default=64)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--taus", default="0.01,0.05,0.1,0.2")
    p.add_argument("--rank-frac", type=float, default=0.25)
    p.add_argument("--methods", default="svd,lora,geom")
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--ft-steps", type=int, default=200)
    p.add_argument("--ft-lr", type=float, default=1e-4)
    p.add_argument("--lora-ratio", type=float, default=16.0)
    p.add_argument("--spectrum", default=None)
    args = p.parse_args()
    model, tok = setup(args)
    spec_path = args.spectrum or os.path.join(os.path.dirname(args.out), f"spectrum_{args.name}_{args.dataset}.json")
    if os.path.exists(spec_path):
        spec = json.load(open(spec_path))
    else:
        spec = spectrum(args, model, tok)
        save_json(spec_path, spec)

    n_layers = len(get_layers(model))
    rank = max(1, int(args.rank_frac * full_rank(model)))
    total = rank * n_layers
    eval_seqs = load_sequences(tok, args.dataset, args.n_eval, args.seq_len, args.seed, skip=args.n)
    ft_seqs = load_sequences(tok, args.dataset, args.n, args.seq_len, args.seed)
    base_ppl = perplexity(model, eval_seqs, args.device)
    del model
    torch.cuda.empty_cache()

    rows = []
    taus = [float(x) for x in args.taus.split(",")]
    for method in args.methods.split(","):
        settings = taus if method == "geom" else [None]
        for tau in settings:
            if method == "geom":
                ks = [rank_for_tau(torch.tensor(e), tr, tau) for e, tr in zip(spec["eigs"], spec["trace"])]
                ranks = match_budget(ks, total)
            else:
                ks = None
                ranks = [rank] * n_layers
            m, _ = load_model(args.model, args.dtype, args.device)
            n_params = compress(get_layers(m), ranks)
            if method == "lora":
                finetune(m, ft_seqs, args.ft_steps, args.ft_lr, args.lora_ratio, args.device)
            ppl = perplexity(m, eval_seqs, args.device)
            rows.append(dict(
                model=args.name, dataset=args.dataset, method=method, tau=tau,
                uniform_rank=rank, ranks=" ".join(map(str, ranks)),
                k_tau="" if ks is None else " ".join(map(str, ks)),
                lowrank_params=n_params, ppl_base=base_ppl, ppl=ppl, delta_ppl=ppl - base_ppl,
            ))
            print(rows[-1]["method"], tau, ppl)
            del m
            torch.cuda.empty_cache()
            save_rows(args.out, rows)


if __name__ == "__main__":
    main()
