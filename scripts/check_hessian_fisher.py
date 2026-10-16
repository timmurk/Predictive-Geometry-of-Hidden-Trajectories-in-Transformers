import torch
from torch.func import grad, jvp

from pgeom.common import base_parser, setup, parse_layers, examples, save_rows
from pgeom.geometry import (PullbackFisher, rand_dir, unit, top_eig, project_null,
                            kl_to, ce, fd2)
from pgeom.model import Trace


def main():
    p = base_parser()
    p.set_defaults(dtype="fp32", n=3)
    p.add_argument("--rho", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n-top", type=int, default=1)
    p.add_argument("--n-null", type=int, default=2)
    p.add_argument("--n-rand", type=int, default=2)
    args = p.parse_args()
    model, tok = setup(args)
    rows = []
    for i, ids, t in examples(args, tok):
        y = ids[0, t + 1 : t + 2]
        tr = Trace(model, ids[:, : t + 1])
        for layer in parse_layers(args.layers, tr.n_layers):
            x = tr.hidden(layer)
            f = tr.downstream(layer, [t])
            op = PullbackFisher(f, x)
            vals, basis = top_eig(op, args.k)
            onehot = torch.zeros_like(op.p)
            onehot[0, y] = 1
            g = op.jtw(op.p - onehot)

            def ce_fn(h):
                return -torch.log_softmax(f(h), -1)[0, y[0]]

            def kl_loss(h):
                with torch.no_grad():
                    return kl_to(op.p, f(h))

            def ce_loss(h):
                with torch.no_grad():
                    return ce(f(h), y)

            dirs = [("top_fisher", basis[:, j].view_as(x)) for j in range(args.n_top)]
            dirs.append(("grad_aligned", unit(g)))
            dirs += [("projected_null", unit(project_null(rand_dir(x), basis))) for _ in range(args.n_null)]
            dirs += [("random", rand_dir(x)) for _ in range(args.n_rand)]
            a = args.rho * x.float().norm().item()
            for kind, v in dirs:
                kv = op.mv(v)
                hv = jvp(grad(ce_fn), (x,), (v.to(x.dtype),))[1].float()
                vkv = (v * kv).sum().item()
                vhv = (v * hv).sum().item()
                rows.append(dict(
                    model=args.name, dataset=args.dataset, example=i, layer=layer,
                    direction_type=kind, vKv_fisher=vkv, vHv_CE=vhv,
                    CE_minus_K_rel_error=abs(vhv - vkv) / max(abs(vhv), 1e-30),
                    cos_HCEv_Kv=torch.nn.functional.cosine_similarity(hv.flatten(), kv.flatten(), 0).item(),
                    teacherKL_fd2=fd2(kl_loss, x, v, a), CE_fd2=fd2(ce_loss, x, v, a),
                ))
            del op
        print(args.name, "example", i, "done")
        save_rows(args.out, rows)


if __name__ == "__main__":
    main()
