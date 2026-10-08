# LoRA vs KronA — fast Unsloth fine-tune (Colab)

Fine-tune a small chat model **twice** — once with **LoRA** and once with **KronA** — then
compare them apples-to-apples. Everything runs on **Unsloth** in a single **Google Colab**
notebook (free T4 GPU) and finishes in minutes.

> **KRONA = KronA** = the Kronecker adapter ([arXiv:2212.10650](https://arxiv.org/abs/2212.10650)).
> In HuggingFace PEFT this is the **`LoKr`** adapter (`LoKrConfig`) — that is what we train.

> **Why Unsloth, and why the adapters are attached by hand:** Unsloth has **no `LoKr`
> (KronA) support** — its `get_peft_model` only builds a `LoraConfig`. So Unsloth loads and
> patches the base model (fast kernels, 4-bit, memory-safe gradient checkpointing) and then
> **both** adapters are attached with plain `peft.get_peft_model()`. Both arms therefore get
> the exact same speedups, and only the adapter type differs.

---

## What you get

- `notebooks/lora_krona_gemma4_e2b.ipynb` — the whole pipeline in **5 code cells**
  (install+login → load → dataset → train both → compare & save).
- A comparison table: trainable params, peak VRAM, train time, eval loss, perplexity.
- Side-by-side generated samples from both adapters, and both saved adapters zipped.

There are **no Python helper files** — all code lives in the notebook.

---

## Requirements

| | |
|---|---|
| Compute | Colab, GPU runtime (**T4, 16 GB**) |
| Model | `unsloth/gemma-3-1b-it` — small, text-only, **public** (not gated) |
| Dataset | `databricks/databricks-dolly-15k` (capped) |
| Precision | 4-bit (nf4) + fp16 compute, via Unsloth |
| Hugging Face | A **read** access token |

The model was deliberately **downshifted from Gemma 4 E2B to Gemma 3 1B** so a full
LoRA-vs-KronA run is fast and reliable: it is not gated, needs no particular `transformers`
version, and fits the T4 with room for eval and generation.

Dependencies are installed **inside the notebook** (`pip install -q unsloth hf_transfer`) —
Unsloth pins its own compatible `transformers`/`peft`/`trl`/`bitsandbytes`. There is no
`requirements.txt` to keep in sync.

---

## Before you run (once)

1. **Use a GPU runtime.** In Colab: `Runtime → Change runtime type → T4 GPU`.
2. **Create a read token.** Hugging Face → *Settings → Access Tokens → New token (read)*.
   No license acceptance is needed — the model is public.

The notebook reads the Colab secret `HF_TOKEN` if present, otherwise it prompts. The token
is never stored in the file.

---

## How to run

1. Open `notebooks/lora_krona_gemma4_e2b.ipynb` in Colab
   (or upload it: *File → Upload notebook*).
2. Run the cells **top to bottom**:

   | Step | Cell |
   |---|---|
   | 1 | Install Unsloth + log in to Hugging Face |
   | 2 | Config + load the 4-bit base model with Unsloth |
   | 3 | Build the chat dataset |
   | 4 | **Train A (LoRA) and B (KronA)** |
   | 5 | Compare, show samples, save the adapters |

3. Read the comparison table at the end.

A full run takes roughly **3–8 min** on a T4 with the defaults.

---

## Knobs

All in the `CFG` dict at the top of cell 2, so a run stays fast:

| Knob | Meaning | Default |
|---|---|---|
| `model` | Base model | `unsloth/gemma-3-1b-it` |
| `max_samples` | Rows used from the dataset | 1000 |
| `eval_size` | Rows held out for evaluation | 64 |
| `max_seq` | Max sequence length | 512 |
| `steps` | Training steps per method | 60 |
| `bs` | Per-device batch size | 2 |
| `accum` | Gradient accumulation | 4 |
| `lr` | Shared learning rate | 2e-4 |
| `r` / `alpha` | Adapter rank / alpha (same for both) | 16 / 32 |
| `seed` | Seed | 42 |

Raise `max_samples` / `steps` for a more meaningful result.

---

## Fair-comparison rules

Both runs use the **same** data, seed, steps, learning rate, effective batch size,
precision and target modules. **Only the adapter type differs** (LoRA vs KronA), so any
difference in the results is attributable to the method.

---

## Notes & limitations

- Gemma 3 1B is small — 60 steps is a **smoke test** of the two adapters, not a scientific
  comparison. Scale the knobs up for a meaningful result.
- `dolly-15k` is a small, general-purpose instruction set; it is a comparison vehicle,
  not a production dataset.
- Gemma 3 is natively multimodal; this project uses the **text** path only.
- Want the original (slower) target instead? Set `model = "unsloth/gemma-4-E2B-it"`.
  Unsloth supports Gemma 4 only on transformers ≥ 5.5, so that is a riskier run.
- Related work worth a look: **Kron-LoRA** ([arXiv:2508.01961](https://arxiv.org/abs/2508.01961)),
  a hybrid Kronecker + LoRA adapter.

---

## Repository layout

```text
plan.md                                   task checklist
README.md                                 this file
notebooks/lora_krona_gemma4_e2b.ipynb      the runnable notebook
memories/memory.md                        project notes
```
