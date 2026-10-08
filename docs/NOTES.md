# Project notes, gotchas and decisions

Read this before changing the training code — most of it was learned the hard way on a
real T4 run.

---

## The one hard constraint: KronA needs 16-bit

`load_in_4bit = False` is **required**, not a preference.

PEFT's `LoKrLayer.get_delta_weight` ends with:

```python
weight = make_kron(w1, w2, self.scaling[adapter_name])
weight = weight.reshape(base_layer.weight.shape)   # <-- explodes on 4-bit
```

On a **bitsandbytes 4-bit** layer the parameter is a `Params4bit`, so
`base_layer.weight.shape` reports the **nibble-packed storage** rather than the logical
out/in shape. For Gemma 3 1B's `q_proj` (`Linear(in=1152, out=1024)` = 1,179,648
elements) that gives:

```
RuntimeError: shape '[589824, 1]' is invalid for input of size 1179648
```

**589824 is exactly half of 1179648** — the 4-bit packing. `torch.kron` produces the
logical 1,179,648 elements, which cannot reshape to the packed shape.

Corroborating evidence from a real run:

```
WARNING:unsloth.import_fixes: Unsloth: could not restore MoE Linear LoRA targets:
'LoKrConfig' object has no attribute 'target_parameters'
```

Unsloth's PEFT integration assumes LoRA, and LoKr doesn't carry LoRA's fields.

**Consequence: `LoKr` is not QLoRA-compatible.** Any LoRA-vs-KronA comparison on a
4-bit-only budget is impossible unless you run LoRA in 4-bit and KronA in 16-bit, which
breaks the fairness premise. We run **both** arms in 16-bit.

---

## `LoraConfig` and `LoKrConfig` do not share a shape

Never build them from one shared kwargs dict — this caused a real
`TypeError: LoraConfig.__init__() got an unexpected keyword argument 'alpha'`.

| | `LoraConfig` | `LoKrConfig` |
|---|---|---|
| rank | `r` | `r` |
| scaling | **`lora_alpha`** | **`alpha`** |
| dropout | `lora_dropout` | `rank_dropout` |
| `bias` | ✅ `"none"` / `"all"` / `"lora_only"` | ❌ **field does not exist** |

`LoKrConfig` → `LycorisConfig` → `PeftConfig`; `PeftConfig` has no `bias` at all, so
`bias="none"` fails on the KronA arm. See `krona/adapters.py`.

---

## Why Unsloth loads the model but plain PEFT attaches the adapters

Unsloth does the heavy lifting (fast kernels, memory-safe setup, gradient checkpointing)
and **fixes the OOM** that the earlier plain `transformers` + `peft` version hit: PEFT's
`prepare_model_for_kbit_training` upcasts every non-quantised parameter to fp32, which
was a single ~8.75 GiB allocation on a T4.

But Unsloth has **no `LoKr` support** — its `get_peft_model` only builds a `LoraConfig`.
So the base model is loaded *through Unsloth* and **both** adapters are attached with
plain `peft.get_peft_model()`. Both arms therefore get identical speedups and only the
adapter type differs.

Do **not** use `FastLanguageModel.get_peft_model` for the LoRA arm only — that would give
LoRA Unsloth's fused kernels and KronA nothing, contaminating the comparison.

---

## fp16 is forbidden for Gemma

Unsloth prints `Using float16 precision for gemma3 won't work! Using float32.` Gemma
overflows in fp16. Never hardcode `fp16=True`; we pass
`bf16=torch.cuda.is_bf16_supported()`, which is `False` on a T4 (→ fp32) and `True` on
Ampere+ (→ bf16, faster and more stable).

---

## Dataset formatting

Rows are pre-rendered with `tokenizer.apply_chat_template(..., tokenize=False)` into a
plain **`text`** column. A pre-rendered string is accepted by every TRL version, whereas
a `messages` column depends on version-specific auto-detection.

---

## Layout & the single code path

All logic lives in the `krona/` package. There is exactly **one** implementation:

```
run.py                  -> krona.pipeline.run(cfg)
lora_krona_gemma4_e2b.ipynb (2 cells) -> krona.pipeline.run(cfg)
```

`krona/pipeline.py::run()` does environment check -> dataset -> both arms -> report. Neither
entry point contains logic of its own, so the notebook and the CLI cannot drift apart.

---

## Colab vs RunPod vs local

The code is runtime-agnostic. `krona/env.py` detects the environment and
`Config.out_dir` defaults to `/workspace/outputs` on RunPod (a network volume survives a
pod restart) and `outputs/` elsewhere.

| | Colab T4 | RunPod (Ampere+) |
|---|---|---|
| bf16 | ❌ → fp32 | ✅ |
| VRAM | 16 GB | 24 GB+ |
| Suggested model | Gemma 3 **1B** | Gemma 3 **4B** |
| Persistence | ❌ (VM wiped) | ✅ with a network volume |

Same 16-bit constraint on both. On a bigger GPU, raise `model` rather than lowering
precision.

---

## Artifacts

Keep `results.json` and the adapter folders. They are small; the pod is not.
`python compare.py <a>/results.json <b>/results.json` diffs two runs *and* reports which
config keys differed, so a comparison can't silently be apples-to-oranges.

---

## Known sharp edges elsewhere

- Unsloth installs its own pinned `transformers`/`peft`/`trl`/`bitsandbytes`. Do not pin
  them in `requirements.txt` — that fights Unsloth's resolver.
- `sft_config()` filters kwargs against `dataclasses.fields(SFTConfig)`, so a TRL rename
  degrades gracefully instead of crashing.
- On an unusual CUDA/PyTorch combo, let Unsloth pick the build tag:
  `wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -`
