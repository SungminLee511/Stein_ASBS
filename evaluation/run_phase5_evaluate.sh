#!/bin/bash
# Phase 5: Comprehensive evaluation of all trained experiments
set -e

echo "=== Phase 5: Comprehensive Evaluation ==="

python evaluate_all.py \
  --outputs_root outputs \
  --results_dir results \
  --n_samples 2000 \
  --n_eval_seeds 5

echo "=== Phase 5 Complete ==="

echo "=== Generating Results Report ==="
python generate_results.py \
  --results_dir results \
  --output RESULTS.md

echo "=== Done. Check RESULTS.md ==="
