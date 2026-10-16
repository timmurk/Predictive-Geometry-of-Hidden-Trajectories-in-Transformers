import torch
import torch.nn.functional as F

from .geometry import fisher_mv
from .model import final_logits


def kd_loss(zs, zt, kind, temp=2.0, alpha=0.1):
    ls = F.log_softmax(zs / temp, -1)
    lt = F.log_softmax(zt / temp, -1)
    if kind == "fkl":
        loss = (lt.exp() * (lt - ls)).sum(-1)
    elif kind == "rkl":
        loss = (ls.exp() * (ls - lt)).sum(-1)
    elif kind == "skewkl":
        mix = torch.log(alpha * lt.exp() + (1 - alpha) * ls.exp() + 1e-12)
        loss = (lt.exp() * (lt - mix)).sum(-1)
    else:
        raise ValueError(kind)
    return loss.mean() * temp * temp


def geo_loss(teacher, hs, ht):
    zt = final_logits(teacher, ht).float()
    zs = final_logits(teacher, hs).float()
    dz = zs - zt
    p = torch.softmax(zt.detach(), -1)
    return 0.5 * (dz * fisher_mv(p, dz)).sum(-1).mean()
