# SDR-ASBS → Vanilla ASBS Fine-tuning

## Problem
SDR-ASBS improves mode coverage (Mode TV, W2, Sinkhorn) but degrades Energy W2.
The KSD repulsive gradient pushes particles apart, including away from mode centers,
causing intra-mode energy distribution mismatch.

## Idea
Two-phase curriculum training:
1. **Phase 1 (SDR-ASBS):** Train with KSD regularization to discover and cover all modes
2. **Phase 2 (Vanilla ASBS):** Resume from SDR checkpoint with lambda=0, let particles relax to correct energy levels within each mode

## Why it should work
- Mode coverage is a "structural" property encoded in the network weights — it persists after removing KSD
- Energy misplacement is caused by the repulsive force — removing it lets the ASBS loss pull particles back to correct positions
- Essentially: SDR for exploration, vanilla for exploitation

## Risks
- Too many fine-tune epochs → mode collapse back to vanilla ASBS failure mode
- Need to monitor Mode TV + Energy W2 jointly and find the sweet spot
- Learning rate sensitivity — too high could destabilize coverage quickly
