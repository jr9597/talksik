#!/bin/bash
# Download the four Saarbrücken zips, train the encoder, and deploy the site.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP_DIR="$ROOT/data/zips"
LOG_DIR="$ROOT/data/logs"
mkdir -p "$ZIP_DIR" "$LOG_DIR"
cd "$ZIP_DIR"

fetch() {
  local out="$1" url="$2" size="$3"
  while true; do
    local have
    have=$(stat -f%z "$out" 2>/dev/null || echo 0)
    if [ "$have" -ge "$size" ]; then
      echo "done $out"
      break
    fi
    echo "resume $out ($have / $size)"
    curl -L --fail --retry 5 --retry-all-errors -C - \
      --connect-timeout 20 --speed-limit 20000 --speed-time 30 \
      -o "$out" "$url" >> "$LOG_DIR/${out}.log" 2>&1 || true
    sleep 3
  done
}

fetch "healthy.zip" "https://zenodo.org/api/records/16874898/files/healthy.zip/content" 6020018616 &
fetch "Hyperfunktionelle Dysphonie.zip" "https://zenodo.org/api/records/16874898/files/Hyperfunktionelle%20Dysphonie.zip/content" 1928321725 &
fetch "Rekurrensparese.zip" "https://zenodo.org/api/records/16874898/files/Rekurrensparese.zip/content" 1900446365 &
fetch "Laryngitis.zip" "https://zenodo.org/api/records/16874898/files/Laryngitis.zip/content" 1193637707 &
wait
echo "all zips complete"

cd "$ROOT"
uv run python train/prepare.py
uv run python train/hubert_probe.py
uv run python train/train.py
wrangler pages deploy web --project-name talksik --commit-dirty=true
echo "deployed"
