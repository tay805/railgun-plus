#!/usr/bin/env bash
# Build LaCAM ONCE on Kaggle, then save the binary as a Kaggle dataset so you
# never recompile. Run this in a Kaggle cell with:  !bash scripts/build_lacam.sh
#
# After it finishes, the binary is at /kaggle/working/lacam/build/main
# (path may vary by fork — the script prints the exact path it found).
#
# THEN: download that binary, create a Kaggle Dataset containing it, and in
# future notebooks load it from /kaggle/input/<your-lacam-dataset>/ instead of
# rebuilding. That makes LaCAM as stable as a pip package.
set -e

cd /kaggle/working
echo "=== cloning LaCAM ==="
# Okumura's reference repo. (LaCAM2 / lacam3 also work; adjust if you prefer.)
rm -rf lacam
git clone --recursive https://github.com/Kei18/lacam.git
cd lacam

echo "=== building (cmake + make) ==="
cmake -B build -DCMAKE_BUILD_TYPE=Release
make -C build -j4

echo "=== locating the built binary ==="
# common locations across forks
for cand in build/main build/lacam build/app/main; do
  if [ -x "$cand" ]; then
    echo "FOUND LaCAM binary at: /kaggle/working/lacam/$cand"
    "$cand" --help 2>&1 | head -20 || true
    echo ""
    echo "Save THIS file as a Kaggle dataset and load it in future notebooks."
    exit 0
  fi
done
echo "Build finished but binary not at an expected path. Check build/ contents:"
ls -la build/
