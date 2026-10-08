# Setup, commands and conventions

Quick reference for this repo. Run everything from the **repo root**.

---

## Install

```bash
# Colab / RunPod / local, all the same
pip install -r requirements.txt

# RunPod, one shot (creates .venv, installs, sets up .env):
bash scripts/setup_runpod.sh
```

> `unsloth` pulls in its own pinned `transformers`/`peft`/`trl`/`bitsandbytes`, so those are
> deliberately *not* pinned in `requirements.txt` — pinning them fights Unsloth's resolver.
> On an unusual CUDA/PyTorch combo let Unsloth choose the build tag:
> `wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -`

---

## Run the experiment

```bash
python run.py                      # defaults + .env
python run.py --steps 200          # override any knob
python run.py --print-config       # show the effective config and exit
python run.py --skip-login         # public repos only
```

Same knobs as `KRONA_*` environment variables:

```bash
KRONA_STEPS=200 KRONA_MODEL=unsloth/gemma-3-4b-it python run.py
```

Outputs land in `./outputs` (or `/workspace/outputs` on RunPod) as `adapter_lora/`,
`adapter_krona/` and `results.json`.

---

## Compare two runs

```bash
python compare.py outputs/results.json
python compare.py run_a/results.json run_b/results.json
```

Diffs the metrics **and** reports which config keys differed, so a comparison can never be
silently apples-to-oranges.

---

## Notebook

`notebooks/lora_krona_gemma4_e2b.ipynb` is a **thin wrapper** over the same package — same
code path as `run.py`, no duplicated logic. It clones the repo if `KRONA_REPO` is set.

---

## Package layout

```
krona/
  config.py     Config dataclass + KRONA_* env overrides
  env.py        Colab/RunPod detection, GPU report, HF login, dependency check
  data.py       dolly-15k -> chat-rendered `text` column
  adapters.py   LoraConfig / LoKrConfig construction (they are NOT interchangeable)
  train.py      model loading, shared training loop, Result
  report.py     comparison table, samples, results.json
run.py          CLI entry point
compare.py      diff two results.json files
scripts/        setup_runpod.sh, pack_outputs.sh
docs/NOTES.md   the hard-won gotchas — READ THIS
```

---

## Persist your work

The pod/VM is **not** persistent. Before stopping:

```bash
bash scripts/pack_outputs.sh              # -> outputs/adapters.zip
bash scripts/pack_outputs.sh /workspace/outputs
```

`results.json` + the adapter folders are the artifacts worth keeping.
