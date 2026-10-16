import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}


def load_model(name, dtype="bf16", device="cuda"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=DTYPES[dtype], attn_implementation="eager"
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def get_layers(model):
    return model.model.layers


def first_out(out):
    return out[0] if isinstance(out, tuple) else out


def final_logits(model, h):
    z = model.lm_head(model.model.norm(h))
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap:
        z = torch.tanh(z / cap) * cap
    return z


class Trace:
    def __init__(self, model, input_ids):
        self.model = model
        self.layers = get_layers(model)
        self.n_layers = len(self.layers)
        self.input_ids = input_ids
        self.inputs = {}
        handles = [
            layer.register_forward_pre_hook(self._grab(i), with_kwargs=True)
            for i, layer in enumerate(self.layers)
        ]
        with torch.no_grad():
            self.logits = model(input_ids=input_ids, use_cache=False).logits.float()
        for h in handles:
            h.remove()

    def _grab(self, i):
        def hook(module, args, kwargs):
            kw = dict(kwargs)
            if args:
                self.inputs[i] = (args[0].detach(), args[1:], kw)
            else:
                self.inputs[i] = (kw.pop("hidden_states").detach(), (), kw)
        return hook

    def hidden(self, layer):
        return self.inputs[layer][0]

    def block(self, i, h):
        _, rest, kw = self.inputs[i]
        return first_out(self.layers[i](h, *rest, **kw))

    def run(self, layer, x, stop=None):
        h = x
        for i in range(layer, self.n_layers if stop is None else stop):
            h = self.block(i, h)
        return h

    def logits_from(self, layer, x, positions):
        h = self.run(layer, x)
        z = final_logits(self.model, h[0, positions])
        return z if z.dtype == torch.float64 else z.float()

    def downstream(self, layer, positions):
        return lambda x: self.logits_from(layer, x, positions)
