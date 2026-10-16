import os
import torch
from datasets import load_dataset


SOURCES = {
    "wikitext": ("wikitext", "wikitext-103-raw-v1", "validation"),
    "openwebtext": (os.environ.get("OWT_PATH", "Skylion007/openwebtext"), None, "train"),
    "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT", "train"),
}


def text_rows(name, seed, split):
    if os.path.isfile(name):
        with open(name) as f:
            for line in f:
                yield {"text": line}
        return
    path, cfg, default_split = SOURCES[name]
    split = split or default_split
    ds = load_dataset(path, cfg, split=split, streaming=True)
    if name != "wikitext" or split == "train":
        ds = ds.shuffle(seed=seed, buffer_size=10000)
    yield from ds


def load_sequences(tok, name, n, seq_len, seed=0, skip=0, split=None):
    buf, out = [], []
    for row in text_rows(name, seed, split):
        text = row["text"]
        if not text.strip():
            continue
        buf.extend(tok(text, add_special_tokens=False)["input_ids"])
        buf.append(tok.eos_token_id)
        while len(buf) >= seq_len:
            out.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(out) >= skip + n:
                return torch.tensor(out[skip:])
    return torch.tensor(out[skip:])


def pick_target(seq_len, rng, margin=1):
    return int(rng.integers(seq_len // 2, seq_len - margin))

