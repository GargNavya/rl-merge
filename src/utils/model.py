import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_dtype(name: str = "auto") -> torch.dtype:
    if name != "auto":
        return getattr(torch, name)
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path: str, dtype: str = "auto"):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=pick_dtype(dtype))
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model
