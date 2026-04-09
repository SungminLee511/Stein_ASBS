"""
scripts/generate_new_benchmark_refs.py

Generate reference samples for the three new benchmarks:
- UnequalGMM (2D, 5 modes)
- MW5 (5D, 32 modes)
- ManyWell32 (32D, 65536 modes)

Also runs basic sanity checks on each energy function.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from adjoint_samplers.energies.new_benchmarks import (
    UnequalGMMEnergy,
    ManyWell5DEnergy,
    ManyWell32DEnergy,
)


def verify_energy(energy, x_test, name):
    """Basic sanity check: eval returns finite, correct shape."""
    E = energy.eval(x_test)
    print(f"\n=== {name} ===")
    print(f"  Input shape: {x_test.shape}")
    print(f"  Energy shape: {E.shape}")
    print(f"  Energy range: [{E.min().item():.4f}, {E.max().item():.4f}]")
    print(f"  All finite: {torch.isfinite(E).all().item()}")

    # Test grad_E (autograd)
    forces = energy(x_test)["forces"]
    print(f"  Forces shape: {forces.shape}")
    print(f"  Forces finite: {torch.isfinite(forces).all().item()}")
    print(f"  Forces norm mean: {forces.norm(dim=-1).mean().item():.4f}")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    n_ref = 10000

    # --- 1. Unequal GMM ---
    print("\n" + "=" * 60)
    print("1. Unequal-Weight GMM (2D, 5 modes)")
    print("=" * 60)

    ugmm = UnequalGMMEnergy(dim=2)

    # Sanity check
    x_test = torch.randn(100, 2)
    verify_energy(ugmm, x_test, "UnequalGMM")

    # Check that energy has two minima near mode centers
    centers = ugmm.get_centers()
    E_centers = ugmm.eval(centers)
    print(f"\n  Mode centers:\n{centers}")
    print(f"  Energy at centers: {E_centers.tolist()}")
    print(f"  Weights: {ugmm.get_weights().tolist()}")

    # Generate reference samples
    print(f"\n  Generating {n_ref} reference samples...")
    samples = ugmm.get_ref_samples(n_ref)
    save_path = os.path.join(data_dir, "test_split_UnequalGMM.npy")
    np.save(save_path, samples.numpy())
    print(f"  Saved to {save_path}")
    print(f"  Sample shape: {samples.shape}")
    print(f"  Sample mean: {samples.mean(dim=0).tolist()}")

    # --- 2. MW5 ---
    print("\n" + "=" * 60)
    print("2. Many-Well MW5 (5D, 32 modes)")
    print("=" * 60)

    mw5 = ManyWell5DEnergy(dim=5)

    # Sanity check
    x_test = torch.randn(100, 5)
    verify_energy(mw5, x_test, "MW5")

    # Check double-well structure
    a_vals = torch.linspace(-3, 3, 100)
    e_1d = a_vals ** 4 - 6 * a_vals ** 2 - 0.5 * a_vals
    left_idx = e_1d[:50].argmin()
    right_idx = e_1d[50:].argmin() + 50
    print(f"\n  1D double-well minima:")
    print(f"    Left well: a={a_vals[left_idx].item():.4f}, E={e_1d[left_idx].item():.4f}")
    print(f"    Right well: a={a_vals[right_idx].item():.4f}, E={e_1d[right_idx].item():.4f}")

    # Mode centers
    mode_centers = mw5.get_mode_centers()
    print(f"  Number of mode centers: {mode_centers.shape[0]}")
    E_modes = mw5.eval(mode_centers)
    print(f"  Energy at mode centers: min={E_modes.min().item():.4f}, max={E_modes.max().item():.4f}")

    # Generate reference samples
    print(f"\n  Generating {n_ref} reference samples (MCMC, may take ~30s)...")
    samples = mw5.get_ref_samples(n_ref)
    save_path = os.path.join(data_dir, "test_split_MW5.npy")
    np.save(save_path, samples.numpy())
    print(f"  Saved to {save_path}")
    print(f"  Sample shape: {samples.shape}")

    # Check bimodality per dimension
    for d in range(5):
        frac_left = (samples[:, d] < 0).float().mean().item()
        print(f"  Dim {d}: frac_left={frac_left:.3f}, frac_right={1-frac_left:.3f}")

    # --- 3. ManyWell32 ---
    print("\n" + "=" * 60)
    print("3. 32D Many-Well (32D, 65536 modes)")
    print("=" * 60)

    mw32 = ManyWell32DEnergy(dim=32)

    # Sanity check
    x_test = torch.randn(100, 32)
    verify_energy(mw32, x_test, "ManyWell32")

    # Check well positions
    left_well, right_well = mw32.get_1d_wells()
    print(f"\n  1D well positions: left={left_well:.4f}, right={right_well:.4f}")

    # Check 2D pair structure: should have 2 wells in a, 1 well in b
    x_test2 = torch.zeros(1, 32)
    print(f"  E(0) = {mw32.eval(x_test2).item():.4f}")

    # Check a-dimension has two minima
    a_vals = torch.linspace(-3, 3, 100)
    x_a_test = torch.zeros(100, 32)
    x_a_test[:, 0] = a_vals
    E_a = mw32.eval(x_a_test)
    a_min_idx = E_a.argmin()
    print(f"  E along first a-dim: min at a={a_vals[a_min_idx].item():.4f}, E={E_a[a_min_idx].item():.4f}")

    # Check b-dimension has single minimum at 0
    b_vals = torch.linspace(-3, 3, 100)
    x_b_test = torch.zeros(100, 32)
    x_b_test[:, 1] = b_vals
    E_b = mw32.eval(x_b_test)
    b_min_idx = E_b.argmin()
    print(f"  E along first b-dim: min at b={b_vals[b_min_idx].item():.4f}, E={E_b[b_min_idx].item():.4f}")

    # Generate reference samples
    print(f"\n  Generating {n_ref} reference samples (MCMC for 16 a-dims, may take ~2min)...")
    samples = mw32.get_ref_samples(n_ref)
    save_path = os.path.join(data_dir, "test_split_ManyWell32.npy")
    np.save(save_path, samples.numpy())
    print(f"  Saved to {save_path}")
    print(f"  Sample shape: {samples.shape}")

    # Check a-dim bimodality for first few pairs
    for pair in range(3):
        a_idx = 2 * pair
        b_idx = 2 * pair + 1
        frac_left_a = (samples[:, a_idx] < 0).float().mean().item()
        b_std = samples[:, b_idx].std().item()
        print(f"  Pair {pair}: a frac_left={frac_left_a:.3f}, b std={b_std:.3f} (expect ~1.0)")

    print("\n" + "=" * 60)
    print("All reference samples generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
