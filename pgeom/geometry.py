import torch
from torch.func import jvp, vjp


def fisher_mv(p, u):
    return p * u - p * (p * u).sum(-1, keepdim=True)


class PullbackFisher:
    def __init__(self, f, x):
        self.f = f
        self.x = x
        z, self.back = vjp(f, x)
        self.z = z.detach()
        self.p = torch.softmax(self.z, -1)

    def jv(self, v):
        return jvp(self.f, (self.x,), (v.to(self.x.dtype),))[1].float()

    def jtw(self, w):
        return self.back(w.to(self.z.dtype))[0].float()

    def mv(self, v):
        return self.jtw(fisher_mv(self.p, self.jv(v)))

    def quad(self, v):
        u = self.jv(v)
        return (u * fisher_mv(self.p, u)).sum().item()


def unit(v):
    return v / v.norm()


def rand_dir(x, mask=None):
    v = torch.randn(x.shape, device=x.device)
    if mask is not None:
        v = v * mask
    return unit(v)


def causal_mask(x, t):
    m = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device)
    m[:, : t + 1] = 1
    return m


def hutchinson(op, m, mask=None):
    s1, s2 = 0.0, 0.0
    for _ in range(m):
        xi = torch.randint(0, 2, op.x.shape, device=op.x.device).float() * 2 - 1
        if mask is not None:
            xi = xi * mask
        y = op.mv(xi)
        s1 += (xi * y).sum().item()
        s2 += (y * y).sum().item()
    tr, tr2 = s1 / m, s2 / m
    return tr, tr2, tr * tr / max(tr2, 1e-30)


def top_eig(op, k, iters=6):
    shape = op.x.shape
    n = op.x.numel()

    def apply(q):
        return torch.stack([op.mv(q[:, i].view(shape)).flatten() for i in range(q.shape[1])], 1)

    q, _ = torch.linalg.qr(torch.randn(n, k, device=op.x.device))
    for _ in range(iters):
        q, _ = torch.linalg.qr(apply(q))
    z = apply(q)
    small = q.T @ z
    vals, vecs = torch.linalg.eigh((small + small.T) / 2)
    order = vals.argsort(descending=True)
    return vals[order].clamp_min(0), q @ vecs[:, order]


def project_null(v, basis):
    flat = v.flatten()
    return (flat - basis @ (basis.T @ flat)).view_as(v)


def project_top(v, basis):
    flat = v.flatten()
    return (basis @ (basis.T @ flat)).view_as(v)


def rank_for_tau(vals, trace, tau):
    cum = torch.cumsum(vals, 0)
    ok = (trace - cum) <= tau * trace
    idx = torch.nonzero(ok)
    return int(idx[0]) + 1 if len(idx) else len(vals)


def token_scores(op, m, kind="fisher"):
    acc = torch.zeros(op.x.shape[1], device=op.x.device)
    p = op.p
    for _ in range(m):
        if kind == "fisher":
            y = torch.multinomial(p, 1)
            u = -p.clone()
            u.scatter_add_(-1, y, torch.ones_like(y, dtype=u.dtype))
        elif kind == "diag":
            u = p.sqrt() * torch.randn_like(p)
        else:
            u = torch.randn_like(p)
        g = op.jtw(u)
        acc += g[0].pow(2).sum(-1)
    return acc / m


def kl_to(p_ref, z):
    p = p_ref.double()
    logq = torch.log_softmax(z.double(), -1)
    return (p * (torch.log(p.clamp_min(1e-300)) - logq)).sum().item()


def ce(z, y):
    return -torch.log_softmax(z.double(), -1).gather(-1, y[:, None]).sum().item()


def shift(x, v, a):
    return (x.double() + a * v.double()).to(x.dtype)


def fd2(loss, x, v, alpha):
    a = alpha
    return (loss(shift(x, v, a)) + loss(shift(x, v, -a)) - 2 * loss(x)) / (a * a)
