# LoRA vs KronA — Fine-tuning with Unsloth (Colab / RunPod / local)

Fine-tune a small chat model **twice** — once with **LoRA** and once with **KronA** — then
compare them apples-to-apples. Runs on **Unsloth**, from the **CLI** or a **notebook**, on
**Colab** or **RunPod**. A whole run takes a few minutes.

> **KRONA = KronA** = the Kronecker adapter ([arXiv:2212.10650](https://arxiv.org/abs/2212.10650)).
> In HuggingFace PEFT this is the **`LoKr`** adapter (`LoKrConfig`) — that is what we train.

---

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

On RunPod:

```bash
bash scripts/setup_runpod.sh
source .venv/bin/activate
python run.py
```

---

## What you get

- A comparison table: trainable params, peak VRAM, train time, eval loss, perplexity.
- Side-by-side generated samples from both adapters.
- Both adapters saved, plus `results.json` for diffing runs later.

| Tool | Purpose |
|---|---|
| `run.py` | run the whole experiment |
| `compare.py` | diff the `results.json` of two runs |
| `notebooks/lora_krona_gemma4_e2b.ipynb` | the same pipeline as a thin notebook wrapper |
| `CHEATSHEET.md` | all commands |
| `docs/NOTES.md` | the hard-won gotchas — **read this before changing the training code** |

---

## Requirements

| | |
|---|---|
| Compute | A GPU: Colab **T4 16 GB**, or RunPod (RTX 3090/4090, A100, …) |
| Model | `unsloth/gemma-3-1b-it` — small, text-only, **public** (not gated) |
| Dataset | `databricks/databricks-dolly-15k` (capped) |
| Precision | **16-bit** — required, see below |
| Hugging Face | Token is **optional** (the default model is public) |

On a bigger GPU, raise the **model** rather than lowering precision:

```bash
KRONA_MODEL=unsloth/gemma-3-4b-it python run.py
```

---

## The one hard constraint: KronA needs 16-bit

`load_in_4bit = False` is required, not a preference.

PEFT's `LoKrLayer.get_delta_weight` ends with `weight.reshape(base_layer.weight.shape)`. On
a **bitsandbytes 4-bit** layer, `weight.shape` reports the *nibble-packed* storage, so the
reshape fails:

```
RuntimeError: shape '[589824, 1]' is invalid for input of size 1179648
```

589824 is exactly half of 1179648 — the 4-bit packing. **`LoKr` is not QLoRA-compatible.**

Running LoRA in 4-bit and KronA in 16-bit would break the fairness premise, so **both arms
run in 16-bit**. Gemma 3 1B fits a T4 comfortably at 16-bit; on a bigger GPU you can move up
to 4B. Full analysis in [`docs/NOTES.md`](docs/NOTES.md).

---

## Why Unsloth loads the model but plain PEFT attaches the adapters

Unsloth gives us the fast kernels and the memory-safe setup (and fixes the fp32-upcast OOM
the earlier plain `transformers` + `peft` version hit on a T4).

But Unsloth has **no `LoKr` support** — its `get_peft_model` only builds a `LoraConfig`. So
the base is loaded *through Unsloth* and **both** adapters are attached with plain
`peft.get_peft_model()`. Both arms get identical speedups; only the adapter type differs.

---

## Knobs

Every field of `krona/config.py` can be set three ways — CLI flag, `KRONA_*` env var, or by
editing the default.

| Knob | Meaning | Default |
|---|---|---|
| `model` | Base model | `unsloth/gemma-3-1b-it` |
| `load_in_4bit` | 4-bit load — **keep `False`**, LoKr breaks on 4-bit | `False` |
| `max_samples` | Rows used from the dataset | 1000 |
| `eval_size` | Rows held out for evaluation | 64 |
| `max_seq` | Max sequence length | 512 |
| `steps` | Training steps per method | 60 |
| `batch_size` | Per-device batch size | 2 |
| `grad_accum` | Gradient accumulation | 4 |
| `lr` | Shared learning rate | 2e-4 |
| `rank` / `alpha` | Adapter rank / alpha (same for both) | 16 / 32 |
| `out_dir` | Outputs (`/workspace/outputs` on RunPod) | auto |
| `seed` | Seed | 42 |

---

## Fair-comparison rules

Both runs use the **same** data, seed, steps, learning rate, effective batch size, precision
and target modules. **Only the adapter type differs**, so any difference is attributable to
the method. `compare.py` also reports which config keys differed between two saved runs.

---

## Notes & limitations

- Gemma 3 1B is small — 60 steps is a **smoke test**, not a scientific comparison. Scale the
  knobs up for a meaningful result.
- **KronA needs 16-bit** (see above), which uses more VRAM than QLoRA would.
- `dolly-15k` is a small, general-purpose instruction set; it is a comparison vehicle, not a
  production dataset.
- Gemma 3 is natively multimodal; this project uses the **text** path only.
- **Save your artifacts.** The pod/VM is ephemeral: `bash scripts/pack_outputs.sh`
- Related work worth a look: **Kron-LoRA**
  ([arXiv:2508.01961](https://arxiv.org/abs/2508.01961)), a hybrid Kronecker + LoRA adapter.

---

## Repository layout

```text
README.md        this file
CHEATSHEET.md    all commands
plan.md          task checklist
docs/NOTES.md    gotchas & decisions (read before editing code)

krona/           the package — all the logic
run.py           CLI entry point
compare.py       diff two results.json

scripts/
  setup_runpod.sh    one-shot RunPod setup
  pack_outputs.sh    zip artifacts for download

notebooks/lora_krona_gemma4_e2b.ipynb   thin wrapper over the package
.env.mock        config template (`.env` is git-ignored)
requirements.txt
memories/memory.md
```
