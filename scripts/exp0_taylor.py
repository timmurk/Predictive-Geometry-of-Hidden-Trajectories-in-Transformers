import torch

from pgeom.common import base_parser, setup, parse_layers, examples, save_rows
from pgeom.geometry import PullbackFisher, rand_dir, kl_to, ce, shift
from pgeom.model import Trace


def main():
    p = base_parser()
    p.add_argument("--radii", default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    p.add_argument("--dirs", type=int, default=8)
    args = p.parse_args()
    model, tok = setup(args)
    radii = [float(r) for r in args.radii.split(",")]
    rows = []
    for i, ids, t in examples(args, tok):
        y = ids[0, t + 1 : t + 2]
        tr = Trace(model, ids[:, : t + 1])
        for layer in parse_layers(args.layers, tr.n_layers):
            x = tr.hidden(layer)
            f = tr.downstream(layer, [t])
            op = PullbackFisher(f, x)
            ce0 = ce(op.z, y)
            onehot = torch.zeros_like(op.p)
            onehot[0, y] = 1
            g = op.jtw(op.p - onehot)
            norm = x.float().norm().item()
            for rho in radii:
                a = rho * norm
                for d in range(args.dirs):
                    v = rand_dir(x)
                    with torch.no_grad():
                        z = f(shift(x, v, a))
                    rows.append(dict(
                        model=args.name, dataset=args.dataset, example=i, layer=layer,
                        rho=rho, dir=d,
                        d_ce=ce(z, y) - ce0,
                        lin=a * (g * v).sum().item(),
                        quad=0.5 * a * a * op.quad(v),
                        d_kl=kl_to(op.p, z),
                    ))
            del op
        print(args.name, "example", i, "done")
        save_rows(args.out, rows)


if __name__ == "__main__":
    main()
