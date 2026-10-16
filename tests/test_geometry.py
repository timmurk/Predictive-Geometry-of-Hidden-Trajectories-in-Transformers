import sys

import torch
from torch.autograd.functional import jacobian
from transformers import AutoModelForCausalLM

from pgeom.geometry import PullbackFisher, top_eig, hutchinson, token_scores, fisher_mv
from pgeom.model import Trace


def load(path):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float64, attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def dense_k(f, x):
    jac = jacobian(f, x).reshape(-1, x.numel())
    p = torch.softmax(f(x), -1).reshape(-1)
    fisher = torch.diag(p) - torch.outer(p, p)
    return jac.T @ fisher @ jac


def check(path):
    torch.manual_seed(0)
    model = load(path)
    ids = torch.randint(3, model.config.vocab_size, (1, 12))
    tr = Trace(model, ids)
    t = 11
    full = model(input_ids=ids).logits[0, t]
    for layer in range(tr.n_layers):
        z = tr.logits_from(layer, tr.hidden(layer), [t])[0]
        assert torch.allclose(z, full.to(z.dtype), atol=1e-8)
    layer = 1
    x = tr.hidden(layer)
    f = tr.downstream(layer, [t])
    op = PullbackFisher(f, x)
    k = dense_k(lambda h: f(h)[0], x).float()
    v = torch.randn(x.shape)
    assert torch.allclose(op.mv(v).flatten(), k @ v.flatten(), rtol=1e-4, atol=1e-6)
    assert abs(op.quad(v) - (v.flatten() @ k @ v.flatten()).item()) < 1e-5 * max(1.0, abs(op.quad(v)))
    vals, basis = top_eig(op, 4, iters=30)
    exact = torch.linalg.eigvalsh(k).flip(0)[:4]
    assert torch.allclose(vals, exact, rtol=2e-2, atol=1e-6), (vals, exact)
    tr1, tr2, _ = hutchinson(op, 400)
    assert abs(tr1 - k.trace().item()) < 0.15 * k.trace().item()
    blocks = k.reshape(x.shape[1], x.shape[2], x.shape[1], x.shape[2])
    kappa = torch.stack([blocks[s, :, s, :].trace() for s in range(x.shape[1])])
    est = token_scores(op, 2000, "fisher")
    assert torch.allclose(est, kappa, rtol=0.2, atol=1e-3 * kappa.max()), (est, kappa)
    print(path, "ok")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        check(path)
