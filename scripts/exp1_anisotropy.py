import torch

from pgeom.common import base_parser, setup, parse_layers, examples, save_rows
from pgeom.geometry import (PullbackFisher, rand_dir, unit, top_eig, project_null,
                            hutchinson, kl_to, shift)
from pgeom.model import Trace


def main():
    p = base_parser()
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--dirs", type=int, default=8)
    p.add_argument("--rho", type=float, default=1e-2)
    args = p.parse_args()
    model, tok = setup(args)
    rows = []
    for i, ids, t in examples(args, tok):
        tr = Trace(model, ids[:, : t + 1])
        for layer in parse_layers(args.layers, tr.n_layers):
            x = tr.hidden(layer)
            f = tr.downstream(layer, [t])
            op = PullbackFisher(f, x)
            vals, basis = top_eig(op, args.k, args.iters)
            trace, trace2, reff = hutchinson(op, args.probes)
            a = args.rho * x.float().norm().item()
            obs, null = [], []
            for _ in range(args.dirs):
                v = unit((basis @ torch.randn(args.k, device=x.device)).view_as(x))
                with torch.no_grad():
                    obs.append(kl_to(op.p, f(shift(x, v, a))))
                v = unit(project_null(rand_dir(x), basis))
                with torch.no_grad():
                    null.append(kl_to(op.p, f(shift(x, v, a))))
            rows.append(dict(
                model=args.name, dataset=args.dataset, example=i, layer=layer, target=t,
                kl_obs=sum(obs) / len(obs), kl_null=sum(null) / len(null),
                trace=trace, trace2=trace2, reff=reff,
                top_eigs=" ".join(f"{v:.4g}" for v in vals.tolist()),
            ))
            del op
        print(args.name, "example", i, "done")
        save_rows(args.out, rows)


if __name__ == "__main__":
    main()
