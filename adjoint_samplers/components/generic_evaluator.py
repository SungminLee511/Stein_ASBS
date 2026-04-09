"""
adjoint_samplers/components/generic_evaluator.py

Evaluators for non-particle-system benchmarks:
- GenericEnergyEvaluator: energy_W2 + Sinkhorn divergence for any BaseEnergy with ref samples
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

        # Sinkhorn divergence (sample-space, regularized W2 approximation)
        print("Computing Sinkhorn divergence...")
        sinkhorn_div = self._compute_sinkhorn(samples, ref_samples)

        result = {
            "energy_w2": energy_w2,
            "sinkhorn_div": sinkhorn_div,
            "mean_energy": gen_energy.mean().item(),
        }

        return result

    def _compute_sinkhorn(self, gen_samples: torch.Tensor,
                          ref_samples: torch.Tensor,
                          reg: float = 0.1,
                          max_samples: int = 2000) -> float:
        """Compute Sinkhorn divergence between generated and reference samples.

        Uses debiased Sinkhorn: S(P,Q) = OT_ε(P,Q) - 0.5*OT_ε(P,P) - 0.5*OT_ε(Q,Q)
        to remove entropic bias. Falls back to biased OT_ε(P,Q) if debiased fails.

        Args:
            gen_samples: (B, D) generated samples
            ref_samples: (B, D) reference samples
            reg: entropic regularization (higher = more smoothing, default 0.1)
            max_samples: subsample both sets if larger (memory/speed)

        Returns:
            Sinkhorn divergence (float, non-negative)
        """
        n_gen = len(gen_samples)
        n_ref = len(ref_samples)

        # Subsample if too large
        if n_gen > max_samples:
            idx = torch.randperm(n_gen)[:max_samples]
            gen_samples = gen_samples[idx]
            n_gen = max_samples
        if n_ref > max_samples:
            idx = torch.randperm(n_ref)[:max_samples]
            ref_samples = ref_samples[idx]
            n_ref = max_samples

        gen_np = gen_samples.cpu().detach().numpy().astype(np.float64)
        ref_np = ref_samples.cpu().detach().numpy().astype(np.float64)

        # Cost matrix (squared Euclidean)
        M = pot.dist(gen_np, ref_np, metric='sqeuclidean')

        # Uniform weights
        a = np.ones(n_gen) / n_gen
        b = np.ones(n_ref) / n_ref

        # Debiased Sinkhorn divergence: OT_ε(P,Q) - 0.5*OT_ε(P,P) - 0.5*OT_ε(Q,Q)
        try:
            ot_pq = pot.sinkhorn2(a, b, M, reg=reg, numItermax=500)[0]

            M_pp = pot.dist(gen_np, gen_np, metric='sqeuclidean')
            ot_pp = pot.sinkhorn2(a, a, M_pp, reg=reg, numItermax=500)[0]

            M_qq = pot.dist(ref_np, ref_np, metric='sqeuclidean')
            ot_qq = pot.sinkhorn2(b, b, M_qq, reg=reg, numItermax=500)[0]

            sinkhorn_div = float(max(ot_pq - 0.5 * ot_pp - 0.5 * ot_qq, 0.0))
        except Exception as e:
            print(f"Debiased Sinkhorn failed ({e}), using biased OT_ε(P,Q)")
            sinkhorn_div = float(pot.sinkhorn2(a, b, M, reg=reg, numItermax=500)[0])

        return sinkhorn_div


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
