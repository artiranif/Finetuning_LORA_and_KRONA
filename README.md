# LoRA vs KronA — Fine-tuning Gemma 4 E2B (Colab)

Fine-tune **`google/gemma-4-E2B-it`** for chat, **twice** — once with **LoRA** and once
with **KronA** — then compare them apples-to-apples.

Everything runs in a single **Google Colab** notebook (free T4 GPU).

> **KRONA = KronA** = the Kronecker adapter ([arXiv:2212.10650](https://arxiv.org/abs/2212.10650)).
> In HuggingFace PEFT this is the **`LoKr`** adapter (`LoKrConfig`) — that is what we train.

---

## What you get

- `notebooks/lora_krona_gemma4_e2b.ipynb` — the whole pipeline, one cell per step.
- A comparison table: trainable params, peak VRAM, train time, eval loss, perplexity.
- Side-by-side generated samples from both adapters.

There are **no Python helper files** — all code lives in the notebook.

---

## Requirements

| | |
|---|---|
| Compute | Colab, GPU runtime (**T4, 16 GB**) |
| Model | `google/gemma-4-E2B-it` *(gated — you must accept the license)* |
| Dataset | `databricks/databricks-dolly-15k` (capped) |
| Precision | 4-bit (nf4 + double quant) + fp16 compute |
| Hugging Face | A **read** access token |

---

## Before you run (once)

1. **Use a GPU runtime.** In Colab: `Runtime → Change runtime type → T4 GPU`.
2. **Accept the Gemma license.** Open [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it)
   and click *Agree and access repository*. Without this the model download fails.
3. **Create a read token.** Hugging Face → *Settings → Access Tokens → New token (read)*.

The notebook asks for the token at runtime (`userdata` / `getpass`) — it is never stored in the file.

---

## How to run

1. Open `notebooks/lora_krona_gemma4_e2b.ipynb` in Colab
   (or upload it: *File → Upload notebook*).
2. Run the cells **top to bottom**:

   | Step | Cell |
   |---|---|
   | 1 | Install libraries + check the GPU |
   | 2 | Log in to Hugging Face |
   | 3 | Load the 4-bit model + tokenizer |
   | 4 | Load and format the dataset |
   | 5 | **Train A — LoRA** |
   | 6 | **Train B — KronA** |
   | 7 | Compare (params, VRAM, time, loss, samples) |
   | 8 | Save both adapters |

3. Read the comparison table at the end.

A full run takes roughly **15–30 min** on a T4 with the default (small) dataset cap.

---

## Knobs

All at the top of the notebook, so a run stays fast:

| Knob | Meaning | Default |
|---|---|---|
| `MAX_SAMPLES` | Rows kept from the dataset | small |
| `MAX_STEPS` | Training steps per method | small |
| `BATCH_SIZE` | Per-device batch size | 1–2 |
| `GRAD_ACCUM` | Gradient accumulation | 8 |
| `LEARNING_RATE` | Shared LR | 2e-4 |
| `SEED` | Seed | 42 |

Raise `MAX_SAMPLES` / `MAX_STEPS` for a more meaningful result.

---

## Fair-comparison rules

Both runs use the **same** data, seed, steps, learning rate, effective batch size,
precision and target modules. **Only the adapter type differs** (LoRA vs KronA), so any
difference in the results is attributable to the method.

---

## Notes & limitations

- Gemma 4 is **multimodal**; this project uses the **text** path only.
- Gemma 4 is a **new architecture** — if loading fails, check the `transformers` version
  printed by the setup cell.
- `dolly-15k` is a small, general-purpose instruction set; it is a comparison vehicle,
  not a production dataset.
- VRAM is tight on a free T4, hence 4-bit + gradient checkpointing + gradient accumulation.
- Related work worth a look: **Kron-LoRA** ([arXiv:2508.01961](https://arxiv.org/abs/2508.01961)),
  a hybrid Kronecker + LoRA adapter.

---

## Repository layout

```text
plan.md                                   task checklist
README.md                                 this file
requirements.txt                          pinned libraries
notebooks/lora_krona_gemma4_e2b.ipynb      the runnable notebook
memories/memory.md                        project notes
```
