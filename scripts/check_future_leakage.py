import torch

from pgeom.common import base_parser, setup, parse_layers, examples, save_rows
from pgeom.geometry import (PullbackFisher, rand_dir, unit, top_eig, project_null,
                            causal_mask, kl_to, shift)
from pgeom.model import Trace


class Masked:
    def __init__(self, op, mask):
        self.op, self.mask, self.x = op, mask, op.x

    def mv(self, v):
        return self.op.mv(v * self.mask) * self.mask


def curvatures(f, p_ref, x, v, a):
    with torch.no_grad():
        zp, zm = f(shift(x, v, a)), f(shift(x, v, -a))
    c = [(kl_to(p_ref[j], zp[j]) + kl_to(p_ref[j], zm[j])) / (a * a) for j in range(p_ref.shape[0])]
    return c[0], sum(c[1:])


def main():
    p = base_parser()
    p.set_defaults(dtype="fp32", n=3, dataset="openwebtext")
    p.add_argument("--rho", type=float, default=1e-3)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n-top", type=int, default=1)
    p.add_argument("--n-null", type=int, default=2)
    p.add_argument("--data-driven", action="store_true")
    args = p.parse_args()
    model, tok = setup(args)
    h = args.horizon
    batch = list(examples(args, tok, margin=h + 1))
    rows = []
    for n, (i, ids, t) in enumerate(batch):
        other = batch[(n + 1) % len(batch)][1]
        tr = Trace(model, ids[:, : t + h + 1])
        tr_other = Trace(model, other[:, : t + h + 1]) if args.data_driven else None
        pos = list(range(t, t + h + 1))
        for layer in parse_layers(args.layers, tr.n_layers):
            x = tr.hidden(layer)
            mask = causal_mask(x, t)
            single = PullbackFisher(tr.downstream(layer, [t]), x)
            horizon = PullbackFisher(tr.downstream(layer, pos), x)
            _, b_single = top_eig(Masked(single, mask), args.k)
            _, b_horizon = top_eig(Masked(horizon, mask), args.k)
            dirs = [("single_target_top", b_single[:, j].view_as(x), None) for j in range(args.n_top)]
            dirs += [("horizon_top", b_horizon[:, j].view_as(x), None) for j in range(args.n_top)]
            for _ in range(args.n_null):
                dirs.append(("single_target_projected_null", unit(project_null(rand_dir(x, mask), b_single) * mask), None))
                dirs.append(("horizon_projected_null", unit(project_null(rand_dir(x, mask), b_horizon) * mask), None))
                dirs.append(("random", rand_dir(x, mask), None))
            if args.data_driven:
                with torch.no_grad():
                    upd = (tr.block(layer, x) - x).float() * mask
                onehot = torch.zeros_like(single.p)
                onehot[0, ids[0, t + 1]] = 1
                raw = [
                    ("layer_update_null", upd),
                    ("pair_difference_null", (x - tr_other.hidden(layer)).float() * mask),
                    ("ce_gradient_null", single.jtw(single.p - onehot) * mask),
                ]
                for kind, u in raw:
                    w = project_null(u, b_single) * mask
                    energy = (w.norm() ** 2 / u.norm() ** 2).item()
                    dirs.append((kind, unit(w), energy))
            a = args.rho * (x.float() * mask).norm().item()
            f = tr.downstream(layer, pos)
            for kind, v, energy in dirs:
                imm, fut = curvatures(f, horizon.p, x, v, a)
                rows.append(dict(
                    model=args.name, dataset=args.dataset, example=i, layer=layer,
                    direction_type=kind, single_target_vKv=single.quad(v),
                    horizon_vKv=horizon.quad(v), immediate_fd2=imm, future_sum_fd2=fut,
                    horizon_sum_fd2=imm + fut, future_to_immediate_ratio=fut / max(imm, 1e-30),
                    null_energy=energy,
                ))
            del single, horizon
        print(args.name, "example", i, "done")
        save_rows(args.out, rows)


if __name__ == "__main__":
    main()
