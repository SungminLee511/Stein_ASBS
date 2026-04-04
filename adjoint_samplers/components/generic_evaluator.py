"""
adjoint_samplers/components/generic_evaluator.py

Evaluators for non-particle-system benchmarks:
- GenericEnergyEvaluator: energy_W2 + KSD for any BaseEnergy with ref samples
- RotatedGMMEvaluator: adds mode coverage metrics
"""

import torch
import ot as pot
import numpy as np
from typing import Dict


class GenericEnergyEvaluator:
    """Evaluator for non-particle energy functions (RotGMM, Muller-Brown, BLogReg).

    Computes energy W2 between generated and reference samples.
    Does NOT compute interatomic or point cloud W2 (those are for particle systems).

    Args:
        energy: any BaseEnergy with get_ref_samples() method
    """
    def __init__(self, energy) -> None:
        self.energy = energy
        assert hasattr(energy, 'get_ref_samples'), \
            f"{type(energy).__name__} must implement get_ref_samples()"
        self.ref_samples = energy.get_ref_samples()

    def __call__(self, samples: torch.Tensor) -> Dict:
        B, D = samples.shape
        assert D == self.energy.dim

        # Subsample reference to match generated count
        idxs = torch.randperm(len(self.ref_samples))[:B]
        ref_samples = self.ref_samples[idxs].to(samples.device)

        # Energy W2
        print("Computing energy W2...")
        gen_energy = self.energy.eval(samples)
        ref_energy = self.energy.eval(ref_samples)
        energy_w2 = pot.emd2_1d(
            ref_energy.cpu().detach().numpy(),
            gen_energy.cpu().detach().numpy()
        ) ** 0.5

        result = {
            "energy_w2": energy_w2,
            "mean_energy": gen_energy.mean().item(),
        }

        return result


class RotatedGMMEvaluator(GenericEnergyEvaluator):
    """Evaluator for RotatedGMMEnergy — adds mode coverage metrics.

    Args:
        energy: RotatedGMMEnergy instance
    """
    def __init__(self, energy) -> None:
        super().__init__(energy)

    def __call__(self, samples: torch.Tensor) -> Dict:
        result = super().__call__(samples)

        # Mode coverage
        if hasattr(self.energy, 'count_modes_covered'):
            coverage = self.energy.count_modes_covered(samples)
            result['n_modes_covered'] = coverage['n_modes_covered']
            result['n_modes_total'] = coverage['n_modes_total']
            result['coverage_fraction'] = coverage['coverage_fraction']

        return result
