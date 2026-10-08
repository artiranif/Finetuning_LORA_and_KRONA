#!/usr/bin/env bash
# Package the run artifacts into a single archive to download before the pod stops.
#
#   bash scripts/pack_outputs.sh                       # packs ./outputs
#   bash scripts/pack_outputs.sh /workspace/outputs    # packs a custom dir
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-outputs}"

if [ ! -d "$OUT" ]; then
  echo "!! no such directory: $OUT" >&2
  exit 1
fi

ZIP="$OUT/adapters.zip"
rm -f "$ZIP"

# -r recurse, -q quiet, -j junk paths is intentionally NOT used (we keep the tree).
( cd "$OUT" && zip -qr "$(basename "$ZIP")" . )

echo "==> wrote $ZIP"
du -h "$ZIP" 2>/dev/null || true
echo
echo "Download it from the RunPod file browser, or:"
echo "    scp <pod>:$PWD/$ZIP ."
