"""LoRA vs KronA (LoKr) fine-tuning experiment, built on Unsloth.

Modules:
    config    typed configuration, loaded from defaults + .env
    env       runtime detection (Colab / RunPod / local) and HF login
    data      dolly-15k -> chat `text` column
    adapters  PEFT adapter construction (LoRA and LoKr do NOT share a config shape)
    train     the shared training loop, run once per adapter
    report    comparison table + side-by-side samples
    pipeline  `run()` — the whole experiment in one call
"""

__version__ = "0.1.0"
