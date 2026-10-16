import argparse
import csv
import json
import os
import random

import numpy as np
import torch

from .data import load_sequences, pick_target
from .model import load_model


def base_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--dataset", default="wikitext")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--layers", default="all")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p


def setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tok = load_model(args.model, args.dtype, args.device)
    if args.name is None:
        args.name = args.model.split("/")[-1]
    return model, tok


def parse_layers(spec, n_layers):
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("every"):
        step = int(spec[5:])
        return list(range(0, n_layers, step))
    return [int(x) for x in spec.split(",") if int(x) < n_layers]


def examples(args, tok, skip=0, margin=1):
    seqs = load_sequences(tok, args.dataset, args.n, args.seq_len, args.seed, skip)
    gen = np.random.default_rng(args.seed + skip)
    for i, ids in enumerate(seqs):
        yield i, ids[None].to(args.device), pick_target(args.seq_len, gen, margin)


def save_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows to {path}")


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
