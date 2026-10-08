"""Validate: python compiles, notebook cells parse & stay minimal, CLI still works."""
import ast, json, pathlib, subprocess, sys

failures = 0

# 1) python files
py = [p for p in pathlib.Path(".").rglob("*.py") if ".venv" not in p.parts]
for p in py:
    try:
        ast.parse(p.read_text(encoding="utf-8"))
    except SyntaxError as e:
        failures += 1
        print(f"PY FAIL {p}:{e.lineno}: {e.msg}")
print(f"[1] {len(py)} python files, {failures} failures")

# 2) notebook at the repo ROOT
nb_path = pathlib.Path("lora_krona_gemma4_e2b.ipynb")
assert nb_path.is_file(), "notebook is not at the repo root!"
assert not pathlib.Path("notebooks").exists(), "old notebooks/ dir still present"
print("[2] notebook exists at repo root; notebooks/ removed")

nb = json.loads(nb_path.read_text(encoding="utf-8"))
codes = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(codes, 1):
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    src = "\n".join(("pass" if ln.lstrip().startswith("!") else ln) for ln in src.splitlines())
    try:
        ast.parse(src)
    except SyntaxError as e:
        failures += 1
        print(f"NB FAIL cell {i}:{e.lineno}: {e.msg}")
print(f"[3] {len(codes)} code cells parse")

# 3) minimality: exactly 2 code cells, no logic duplicated, no repo-setup cell
assert len(codes) == 2, f"expected 2 code cells, found {len(codes)}"
whole = "\n".join("".join(c["source"]) for c in codes)
for forbidden in [
    "git clone", "KRONA_REPO", "find_repo", "sys.path",
    "SFTTrainer(", "LoraConfig(", "LoKrConfig(", "apply_chat_template(",
    "load_dataset(", "login(", "getpass", "check_dependencies",
]:
    if forbidden in whole:
        failures += 1
        print(f"NB FAIL: notebook still contains {forbidden!r}")
print("[4] notebook is minimal: no setup/clone/login/dataset/training logic")

# 4) total notebook size
lines = sum(len(("".join(c["source"])).splitlines()) for c in codes)
print(f"[5] {lines} lines of code across the {len(codes)} cells")

# 5) the notebook's exact call path must work
for cmd, needle in [
    ([sys.executable, "-m", "krona.config"], "load_in_4bit  = False"),
    ([sys.executable, "run.py", "--print-config"], "steps         = 60"),
]:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or needle not in r.stdout:
        failures += 1
        print(f"CLI FAIL {' '.join(cmd[1:])}: rc={r.returncode}\n{r.stdout}\n{r.stderr}")
    else:
        print(f"[6] ok: {' '.join(cmd[1:])}")

# 6) pipeline.run must be importable and wired
r = subprocess.run(
    [sys.executable, "-c", "from krona.pipeline import run; import inspect; print(inspect.signature(run))"],
    capture_output=True, text=True,
)
print(f"[7] krona.pipeline.run{r.stdout.strip() or r.stderr.strip()}")
if r.returncode != 0:
    failures += 1

print("\nFAILURES:", failures)
sys.exit(1 if failures else 0)
