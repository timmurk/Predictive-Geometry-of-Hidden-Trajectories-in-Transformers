import torch
from torch import nn


TARGETS = ["q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj",
           "gate_proj", "up_proj", "gate_up_proj", "down_proj"]


class LowRank(nn.Module):
    def __init__(self, a, b, bias=None):
        super().__init__()
        self.a = nn.Parameter(a)
        self.b = nn.Parameter(b)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        y = (x @ self.b.T) @ self.a.T
        return y if self.bias is None else y + self.bias


def factor(w, r, scale=None):
    m = w.float()
    if scale is not None:
        m = m * scale[None]
    u, s, vh = torch.linalg.svd(m, full_matrices=False)
    r = min(r, s.numel())
    root = s[:r].sqrt()
    a = u[:, :r] * root
    b = root[:, None] * vh[:r]
    if scale is not None:
        b = b / scale[None]
    return a.to(w.dtype), b.to(w.dtype)


def target_linears(layer):
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in TARGETS:
            yield name, mod


def set_module(root, name, new):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def compress(layers, ranks, scales=None):
    n_params = 0
    for i, layer in enumerate(layers):
        for name, mod in list(target_linears(layer)):
            sc = None if scales is None else scales.get((i, name))
            a, b = factor(mod.weight.data, ranks[i], sc)
            bias = None if mod.bias is None else mod.bias.data.clone()
            set_module(layer, name, LowRank(a, b, bias))
            n_params += a.numel() + b.numel()
    return n_params


@torch.no_grad()
def activation_scales(model, layers, batches):
    sums, counts, handles = {}, {}, []
    for i, layer in enumerate(layers):
        for name, mod in target_linears(layer):
            def hook(m, args, out, key=(i, name)):
                x = args[0].float().reshape(-1, args[0].shape[-1])
                sums[key] = sums.get(key, 0) + x.pow(2).sum(0)
                counts[key] = counts.get(key, 0) + x.shape[0]
            handles.append(mod.register_forward_hook(hook))
    for ids in batches:
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    return {k: (sums[k] / counts[k]).sqrt().clamp_min(1e-6) for k in sums}


def match_budget(weights, total):
    w = torch.tensor(weights, dtype=torch.float64)
    ranks = torch.floor(w / w.sum() * total).long().clamp_min(1)
    order = torch.argsort(-(w / w.sum() * total - ranks.double()))
    i = 0
    while ranks.sum() < total:
        ranks[order[i % len(order)]] += 1
        i += 1
    while ranks.sum() > total:
        j = int(torch.argmax(ranks))
        ranks[j] -= 1
    return ranks.tolist()
