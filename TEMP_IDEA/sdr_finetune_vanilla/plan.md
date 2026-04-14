# Experiment Plan: SDR → Vanilla Fine-tuning

## Setup
- **Base checkpoints:** Best SDR-ASBS runs (beta=0.7 for Grid25, best beta for MW5)
- **Fine-tune config:** Same as vanilla ASBS but loading from SDR checkpoint
- **Benchmarks:** Grid25, MW5 (start with Grid25 — fast, 2D, easy to visualize)

## Hyperparameters to sweep
- **Fine-tune epochs:** {50, 100, 200, 500} — find when Energy W2 improves before Mode TV degrades
- **Learning rate:** {original LR, 0.5x, 0.1x} — lower LR for stability
- **Seeds:** 3 per config

## Evaluation
- Track all metrics at each checkpoint: Mode TV, Energy W2, W2, Sinkhorn
- Plot metric trajectories over fine-tune epochs to find the crossover point
- Compare final numbers against both vanilla ASBS and SDR-ASBS baselines

## Implementation steps
1. Write a `finetune.py` script (or modify `train.py` to accept `resume_from` + `experiment` override)
2. Load SDR checkpoint, override config to vanilla ASBS (sdr_lambda=0)
3. Train for N additional epochs, save checkpoints frequently
4. Run eval at each checkpoint, plot metric curves

## Expected outcome
A "best of both worlds" model: SDR-level mode coverage + ASBS-level energy accuracy.

## Alternative: Lambda annealing
Instead of a hard switch, anneal sdr_lambda from current value → 0 over the fine-tune phase.
Smoother transition, potentially more stable. Worth testing as a variant.
