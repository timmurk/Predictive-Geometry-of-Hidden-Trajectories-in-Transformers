import torch

from pgeom.common import base_parser, setup, parse_layers, examples, save_rows
from pgeom.geometry import PullbackFisher, hutchinson, kl_to, fd2
from pgeom.model import Trace


def main():
    p = base_parser()
    p.set_defaults(dtype="fp32", n=3, dataset="openwebtext")
    p.add_argument("--rho", type=float, default=1e-3)
    p.add_argument("--probes", type=int, default=32)
    args = p.parse_args()
    model, tok = setup(args)
    rows = []
    for i, ids, t in examples(args, tok):
        tr = Trace(model, ids[:, : t + 1])
        for layer in parse_layers(args.layers, tr.n_layers):
            x = tr.hidden(layer)
            f = tr.downstream(layer, [t])
            op = PullbackFisher(f, x)

            def kl_loss(h):
                with torch.no_grad():
                    return kl_to(op.p, f(h))

            norm = x.float().norm().item()
            c = []
            for _ in range(args.probes):
                v = torch.randn(x.shape, device=x.device)
                a = args.rho * norm / v.norm().item()
                c.append(fd2(kl_loss, x, v, a))
            c = torch.tensor(c)
            tr1, tr2, reff = hutchinson(op, args.probes)
            rows.append(dict(
                model=args.name, dataset=args.dataset, example=i, layer=layer,
                fd_trace=c.mean().item(), fd_reff=(2 * c.mean() ** 2 / c.var()).item(),
                hutch_trace=tr1, hutch_trace2=tr2, hutch_reff=reff,
            ))
            del op
        print(args.name, "example", i, "done")
        save_rows(args.out, rows)


if __name__ == "__main__":
    main()
