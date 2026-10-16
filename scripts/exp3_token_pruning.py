import torch

from pgeom.common import base_parser, setup, parse_layers, examples, save_rows
from pgeom.geometry import PullbackFisher, token_scores, ce
from pgeom.model import Trace, final_logits

METHODS = ["kappa", "jtj", "diag_fisher", "attention", "gradact", "toplogit", "actnorm", "random"]


def scores(method, op, x, attn, y):
    if method == "kappa":
        return token_scores(op, 16, "fisher")
    if method == "jtj":
        return token_scores(op, 16, "jtj")
    if method == "diag_fisher":
        return token_scores(op, 16, "diag")
    if method == "attention":
        return attn
    if method == "actnorm":
        return x[0].float().norm(dim=-1)
    if method == "random":
        return torch.rand(x.shape[1], device=x.device)
    w = torch.zeros_like(op.p)
    if method == "gradact":
        w = op.p.clone()
        w[0, y] -= 1
        return (op.jtw(w) * x.float())[0].abs().sum(-1)
    if method == "toplogit":
        w[0, op.p[0].argmax()] = 1
        return op.jtw(w)[0].norm(dim=-1)
    raise ValueError(method)


def main():
    p = base_parser()
    p.add_argument("--drops", default="0.05,0.1,0.2,0.3")
    p.add_argument("--methods", default=",".join(METHODS))
    args = p.parse_args()
    model, tok = setup(args)
    drops = [float(d) for d in args.drops.split(",")]
    rows = []
    for i, ids, t in examples(args, tok):
        y = ids[0, t + 1 : t + 2]
        ids = ids[:, : t + 1]
        tr = Trace(model, ids)
        with torch.no_grad():
            attns = model(input_ids=ids, output_attentions=True, use_cache=False).attentions
        for layer in parse_layers(args.layers, tr.n_layers):
            x = tr.hidden(layer)
            op = PullbackFisher(tr.downstream(layer, [t]), x)
            ce0 = ce(op.z, y)
            with torch.no_grad():
                upd = tr.block(layer, x)
            attn = attns[layer][0, :, t].float().mean(0)
            for method in args.methods.split(","):
                s = scores(method, op, x, attn, y).clone()
                s[t] = float("inf")
                order = torch.argsort(s)
                for drop in drops:
                    k = int(round(drop * t))
                    h = upd.clone()
                    idx = order[:k]
                    h[:, idx] = x[:, idx]
                    with torch.no_grad():
                        z = final_logits(model, tr.run(layer + 1, h)[0, [t]]).float()
                    rows.append(dict(
                        model=args.name, dataset=args.dataset, example=i, layer=layer,
                        method=method, drop=drop, n_pruned=k, d_ce=ce(z, y) - ce0,
                    ))
            del op
        print(args.name, "example", i, "done")
        save_rows(args.out, rows)


if __name__ == "__main__":
    main()
