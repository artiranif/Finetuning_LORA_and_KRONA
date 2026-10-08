"""PEFT adapter construction.

LoRA and KronA (LoKr) deliberately live behind one function so the two arms cannot
drift apart -- but they are built with *separate* configs, because PEFT's two config
classes do not share a shape.
"""

from __future__ import annotations

from peft import LoraConfig, LoKrConfig, PeftConfig, get_peft_model

from .config import Config

# Attention + MLP projections, identical for both adapters.
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_peft_config(kind: str, cfg: Config) -> PeftConfig:
    """Build the PEFT config for `kind` ("lora" or "krona").

    The two classes are NOT interchangeable:
        LoraConfig : r, lora_alpha, lora_dropout, bias
        LoKrConfig : r, alpha, rank_dropout      (LycorisConfig has no `bias`)
    Passing `alpha=` to LoraConfig, or `bias=` to LoKrConfig, is a TypeError.
    """
    if kind == "lora":
        return LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            lora_dropout=0.0,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if kind == "krona":
        return LoKrConfig(
            r=cfg.rank,
            alpha=cfg.alpha,
            rank_dropout=0.0,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
    raise ValueError(f"Unknown adapter kind: {kind!r} (expected 'lora' or 'krona')")


def attach_adapter(model, kind: str, cfg: Config):
    """Attach the adapter and return `(peft_model, trainable_params, total_params)`."""
    peft_model = get_peft_model(model, build_peft_config(kind, cfg))
    trainable, total = peft_model.get_nb_trainable_parameters()
    return peft_model, trainable, total
