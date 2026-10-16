import argparse
import math

import torch
import torch.nn.functional as F

from pgeom.common import save_rows
from pgeom.data import load_sequences
from pgeom.kd import kd_loss, geo_loss
from pgeom.lowrank import compress, activation_scales
from pgeom.model import load_model, get_layers


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--init", default="svd", choices=["svd", "asvd"])
    p.add_argument("--kd", default="fkl", choices=["fkl", "rkl", "skewkl"])
    p.add_argument("--geo", action="store_true")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--lambda-ce", type=float, default=0.1)
    p.add_argument("--lambda-geo", type=float, default=0.05)
    p.add_argument("--geo-every", type=int, default=4)
    p.add_argument("--geo-layers", default="4,8,12,16,20")
    p.add_argument("--n-calib", type=int, default=16)
    p.add_argument("--n-eval", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    name = args.name or args.model.split("/")[-1]

    teacher, tok = load_model(args.model, args.dtype, args.device)
    student, _ = load_model(args.model, args.dtype, args.device)
    n_layers = len(get_layers(student))
    geo_layers = [int(x) for x in args.geo_layers.split(",") if int(x) < n_layers]

    train = load_sequences(tok, args.dataset, args.steps, args.seq_len, args.seed, split="train")
    evals = load_sequences(tok, args.dataset, args.n_eval, args.seq_len, args.seed)
    scales = None
    if args.init == "asvd":
        calib = [s[None].to(args.device) for s in train[: args.n_calib]]
        scales = activation_scales(student, get_layers(student), calib)
    n_params = compress(get_layers(student), [args.rank] * n_layers, scales)
    teacher_ppl = perplexity(teacher, evals, args.device)
    init_ppl = perplexity(student, evals, args.device)

    params = [prm for n, prm in student.named_parameters() if n.endswith(".a") or n.endswith(".b")]
    for prm in params:
        prm.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    student.train()
    for step in range(args.steps):
        ids = train[step % len(train)][None].to(args.device)
        use_geo = args.geo and step % args.geo_every == 0
        with torch.no_grad():
            t_out = teacher(input_ids=ids, output_hidden_states=use_geo, use_cache=False)
        s_out = student(input_ids=ids, output_hidden_states=use_geo, use_cache=False)
        zs, zt = s_out.logits[0, :-1].float(), t_out.logits[0, :-1].float()
        loss = kd_loss(zs, zt, args.kd, args.temp, args.alpha)
        loss = loss + args.lambda_ce * F.cross_entropy(zs, ids[0, 1:])
        if use_geo:
            g = sum(geo_loss(teacher, s_out.hidden_states[l], t_out.hidden_states[l]) for l in geo_layers)
            loss = loss + args.lambda_geo * g / len(geo_layers)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        if step % 100 == 0:
            print(name, args.init, args.kd, "geo" if args.geo else "", step, round(loss.item(), 4))
    student.eval()
    final_ppl = perplexity(student, evals, args.device)

    method = f"{args.init}+{args.kd}" + ("_geom" if args.geo else "")
    save_rows(args.out, [dict(
        model=name, method=method, init=args.init, kd=args.kd, geo=int(args.geo),
        rank=args.rank, seed=args.seed, steps=args.steps, lowrank_params=n_params,
        teacher_ppl=teacher_ppl, init_ppl=init_ppl, final_ppl=final_ppl,
        delta_teacher=final_ppl - teacher_ppl, delta_init=init_ppl - final_ppl,
    )])


if __name__ == "__main__":
    main()
