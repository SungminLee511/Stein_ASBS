"""
eval_lj38_no_ref.py — Evaluate LJ38 ASBS without ground-truth reference samples.

NOTE on energy convention:
  bgflow's distances_from_vectors returns ALL directed pairs (38×37 = 1406),
  not unique pairs (703). So the LJ energy is 2× the physical single-counted
  value. The oscillator term is NOT affected (computed independently).

  Code convention:  E_code = 2 * E_LJ_physical + E_oscillator
  Physical:         E_phys = E_LJ_physical + E_oscillator

  Known LJ38 global min (single-counted): -173.928 ε
  In code convention (double-counted LJ): -347.856 ε + E_osc ≈ -300

  This script reports BOTH conventions for clarity.

Metrics:
  Tier 1 (intrinsic):
    - Energy stats (code convention + physical LJ decomposition)
    - KSD² — consistency with target score
    - Radius of gyration (rotation-invariant structural metric)
    - Interatomic distance statistics

  Tier 2 (physics-informed):
    - Physical E_LJ vs known LJ38 global minimum
    - Energy histogram (bimodality / double-funnel detection)
    - Coordination number distribution
    - Fraction of samples near known minima

Results saved to evaluation/lj38_no_ref_results.json

Usage:
    cd /home/sky/SML/Stein_ASBS
    CUDA_VISIBLE_DEVICES=0 python -u evaluation/eval_lj38_no_ref.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import datetime
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import hydra

from adjoint_samplers.components.sde import ControlledSDE, sdeint
from adjoint_samplers.components.stein_kernel import compute_ksd_squared, median_bandwidth
from adjoint_samplers.utils.eval_utils import interatomic_dist
from adjoint_samplers.utils.graph_utils import remove_mean
from adjoint_samplers.energies.lennard_jones_energy import lennard_jones_energy_torch
from bgflow.utils import distance_vectors, distances_from_vectors
import adjoint_samplers.utils.train_utils as train_utils


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Evaluation settings
N_SAMPLES = 2000
N_EVAL_SEEDS = 5
BATCH_SIZE = 500

# LJ38 constants
N_PARTICLES = 38
SPATIAL_DIM = 3
DIM = N_PARTICLES * SPATIAL_DIM  # 114
NFE = 1000
SIGMA_MAX = 2
SIGMA_MIN = 0.001
SCALE = 1.5

# Known LJ38 minima (physical single-counted convention, reduced units)
LJ38_PHYSICAL_GLOBAL_MIN = -173.928   # truncated octahedron (FCC)
LJ38_PHYSICAL_ICO_MIN = -173.252      # icosahedral minimum


# ======================================================================
# Energy decomposition
# ======================================================================

@torch.no_grad()
def decompose_energy(samples):
    """
    Decompose energy.eval() into LJ and oscillator components.
    Returns E_lj_double (code convention, 2×), E_osc, E_lj_physical (÷2).
    """
    x_view = samples.view(-1, N_PARTICLES, SPATIAL_DIM)

    # LJ component (double-counted, as in the energy function)
    dists = distances_from_vectors(distance_vectors(x_view))
    lj_pair = lennard_jones_energy_torch(dists, eps=1.0, rm=1.0)
    E_lj_double = lj_pair.view(samples.shape[0], -1).sum(dim=-1)

    # Oscillator component
    x_centered = x_view - x_view.mean(dim=1, keepdim=True)
    E_osc = 0.5 * x_centered.pow(2).sum(dim=(-2, -1))

    # Physical single-counted LJ
    E_lj_physical = E_lj_double / 2.0

    return E_lj_double, E_osc, E_lj_physical


# ======================================================================
# Structural metrics
# ======================================================================

def radius_of_gyration(x, n_particles, spatial_dim):
    """Compute Rg for each sample. x: (B, D) -> (B,) scalar."""
    B = x.shape[0]
    coords = x.view(B, n_particles, spatial_dim)
    com = coords.mean(dim=1, keepdim=True)
    Rg = ((coords - com) ** 2).sum(dim=(1, 2)) / n_particles
    return Rg.sqrt()


def coordination_numbers(x, n_particles, spatial_dim, cutoff=1.5):
    """
    Compute coordination number per atom (avg over samples).
    cutoff=1.5 is standard for LJ (first shell at ~1.0 for rm=1).
    """
    B = x.shape[0]
    coords = x.view(B, n_particles, spatial_dim)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dists = diff.norm(dim=-1)
    mask = torch.eye(n_particles, device=x.device).bool().unsqueeze(0)
    dists = dists.masked_fill(mask, float('inf'))
    coord_nums = (dists < cutoff).float().sum(dim=-1)  # (B, 38)
    return coord_nums.mean().item(), coord_nums.std().item()


def nearest_neighbor_stats(x, n_particles, spatial_dim):
    """Compute nearest-neighbor distance stats."""
    B = x.shape[0]
    coords = x.view(B, n_particles, spatial_dim)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dists = diff.norm(dim=-1)
    mask = torch.eye(n_particles, device=x.device).bool().unsqueeze(0)
    dists = dists.masked_fill(mask, float('inf'))
    nn_dists = dists.min(dim=-1).values
    return {
        'nn_mean': nn_dists.mean().item(),
        'nn_std': nn_dists.std().item(),
        'nn_min': nn_dists.min().item(),
        'nn_max': nn_dists.max().item(),
    }


# ======================================================================
# Energy histogram analysis (on PHYSICAL LJ energies)
# ======================================================================

def energy_histogram_analysis(phys_lj_energies):
    """Analyze physical LJ energy distribution for bimodality hints."""
    E = phys_lj_energies.cpu().numpy()
    result = {}

    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        result[f'phys_lj_p{p}'] = float(np.percentile(E, p))

    # Fraction near known physical minima
    result['frac_phys_lj_below_m140'] = float((E < -140).mean())
    result['frac_phys_lj_below_m160'] = float((E < -160).mean())
    result['frac_phys_lj_below_m170'] = float((E < -170).mean())
    result['frac_phys_lj_below_m173'] = float((E < -173).mean())

    # Skewness and kurtosis
    mu, sigma = E.mean(), E.std()
    if sigma > 0:
        z = (E - mu) / sigma
        result['skewness'] = float((z ** 3).mean())
        result['kurtosis'] = float((z ** 4).mean())
    else:
        result['skewness'] = 0.0
        result['kurtosis'] = 0.0

    # Histogram (for plotting)
    counts, bin_edges = np.histogram(E, bins=50)
    result['hist_counts'] = counts.tolist()
    result['hist_bin_edges'] = bin_edges.tolist()

    return result


# ======================================================================
# Full metric computation
# ======================================================================

@torch.no_grad()
def compute_metrics(samples, energy):
    """Compute all reference-free metrics for a set of LJ38 samples."""
    metrics = {}
    device = samples.device

    # Filter NaN/Inf
    valid_mask = ~(torch.isnan(samples).any(dim=1) | torch.isinf(samples).any(dim=1))
    valid_samples = samples[valid_mask]
    metrics['valid_count'] = int(valid_samples.shape[0])
    metrics['nan_count'] = int((~valid_mask).sum().item())

    if valid_samples.shape[0] < 10:
        print(f"  WARNING: Only {valid_samples.shape[0]} valid samples!", flush=True)
        return metrics

    # ---- Energy decomposition ----
    E_total = energy.eval(valid_samples)
    E_lj_double, E_osc, E_lj_phys = decompose_energy(valid_samples)

    # Filter bad energies
    e_valid = ~(torch.isnan(E_total) | torch.isinf(E_total))
    E_total = E_total[e_valid]
    E_lj_double = E_lj_double[e_valid]
    E_osc = E_osc[e_valid]
    E_lj_phys = E_lj_phys[e_valid]
    valid_samples_clean = valid_samples[e_valid]

    # Code-convention energy (E_total = 2*E_LJ + E_osc)
    metrics['E_total_mean'] = E_total.mean().item()
    metrics['E_total_std'] = E_total.std().item()
    metrics['E_total_min'] = E_total.min().item()
    metrics['E_total_max'] = E_total.max().item()

    # Oscillator energy
    metrics['E_osc_mean'] = E_osc.mean().item()
    metrics['E_osc_std'] = E_osc.std().item()

    # Physical LJ energy (single-counted, the meaningful one)
    metrics['E_lj_phys_mean'] = E_lj_phys.mean().item()
    metrics['E_lj_phys_std'] = E_lj_phys.std().item()
    metrics['E_lj_phys_min'] = E_lj_phys.min().item()
    metrics['E_lj_phys_max'] = E_lj_phys.max().item()
    metrics['E_lj_phys_median'] = E_lj_phys.median().item()

    # Gap to known global minimum (physical convention)
    metrics['gap_to_global_min'] = metrics['E_lj_phys_min'] - LJ38_PHYSICAL_GLOBAL_MIN

    # Energy histogram on physical LJ
    hist_info = energy_histogram_analysis(E_lj_phys)
    metrics.update(hist_info)

    # ---- KSD² (uses code-convention energy, which is what the model was trained on) ----
    try:
        with torch.enable_grad():
            s_req = valid_samples_clean[:min(2000, len(valid_samples_clean))].detach().requires_grad_(True)
            E_req = energy.eval(s_req)
            scores = -torch.autograd.grad(E_req.sum(), s_req)[0]
            s_req = s_req.detach()

        ell = median_bandwidth(s_req)
        metrics['ksd_squared'] = float(compute_ksd_squared(s_req, scores.detach(), ell).item())
        metrics['bandwidth'] = ell.item()
    except Exception as e:
        print(f"  KSD computation failed: {e}", flush=True)
        metrics['ksd_squared'] = float('nan')
        metrics['bandwidth'] = float('nan')

    # ---- Radius of gyration ----
    try:
        rg = radius_of_gyration(valid_samples_clean, N_PARTICLES, SPATIAL_DIM)
        metrics['rg_mean'] = rg.mean().item()
        metrics['rg_std'] = rg.std().item()
    except Exception as e:
        print(f"  Rg computation failed: {e}", flush=True)

    # ---- Interatomic distance stats ----
    try:
        iad = interatomic_dist(valid_samples_clean, N_PARTICLES, SPATIAL_DIM)
        metrics['iad_mean'] = iad.mean().item()
        metrics['iad_std'] = iad.std().item()
    except Exception as e:
        print(f"  IAD computation failed: {e}", flush=True)

    # ---- Nearest-neighbor stats ----
    try:
        nn_stats = nearest_neighbor_stats(valid_samples_clean, N_PARTICLES, SPATIAL_DIM)
        metrics.update(nn_stats)
    except Exception as e:
        print(f"  NN stats failed: {e}", flush=True)

    # ---- Coordination number ----
    try:
        mean_cn, std_cn = coordination_numbers(valid_samples_clean, N_PARTICLES, SPATIAL_DIM)
        metrics['coord_num_mean'] = mean_cn
        metrics['coord_num_std'] = std_cn
    except Exception as e:
        print(f"  Coordination number failed: {e}", flush=True)

    return metrics


# ======================================================================
# Model loading & sample generation
# ======================================================================

def load_model(exp_dir, ckpt_name, device):
    """Load ASBS model from checkpoint."""
    ckpt_path = exp_dir / 'checkpoints' / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    energy = hydra.utils.instantiate(
        OmegaConf.create({
            '_target_': 'adjoint_samplers.energies.lennard_jones_energy.LennardJonesEnergy',
            'dim': DIM, 'n_particles': N_PARTICLES,
        }),
        device=device,
    )
    source = hydra.utils.instantiate(
        OmegaConf.create({
            '_target_': 'adjoint_samplers.utils.dist_utils.CenteredParticlesHarmonic',
            'n_particles': N_PARTICLES, 'spatial_dim': SPATIAL_DIM, 'scale': SCALE,
        }),
        device=device,
    )
    ref_sde = hydra.utils.instantiate(
        OmegaConf.create({
            '_target_': 'adjoint_samplers.components.sde.GraphVESDE',
            'n_particles': N_PARTICLES, 'spatial_dim': SPATIAL_DIM,
            'sigma_max': SIGMA_MAX, 'sigma_min': SIGMA_MIN,
        })
    ).to(device)
    controller = hydra.utils.instantiate(
        OmegaConf.create({
            '_target_': 'adjoint_samplers.components.model.EGNN_dynamics',
            'n_particles': N_PARTICLES, 'spatial_dim': SPATIAL_DIM,
            'hidden_nf': 128, 'n_layers': 5,
            'act_fn': {'_target_': 'torch.nn.SiLU'},
            'recurrent': True, 'tanh': True, 'attention': True,
            'condition_time': True, 'agg': 'sum',
        })
    ).to(device)

    sde = ControlledSDE(ref_sde, controller).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    controller.load_state_dict(ckpt['controller'])
    controller.eval()

    epoch = ckpt.get('epoch', '?')
    print(f"  Loaded checkpoint: {ckpt_name} (epoch {epoch})", flush=True)
    return sde, source, energy, epoch


@torch.no_grad()
def generate_samples(sde, source, n_samples, batch_size, device):
    """Generate terminal samples in batches."""
    ts = train_utils.get_timesteps(t0=0.0, t1=1.0, steps=NFE, rescale_t=None).to(device)
    x1_list = []
    n = 0
    while n < n_samples:
        b = min(batch_size, n_samples - n)
        x0 = source.sample([b]).to(device)
        _, x1 = sdeint(sde, x0, ts, only_boundary=True)
        x1_list.append(x1.cpu())
        n += b
    return torch.cat(x1_list)[:n_samples].to(device)


# ======================================================================
# Main
# ======================================================================

SUMMARY_KEYS = [
    'E_lj_phys_mean', 'E_lj_phys_std', 'E_lj_phys_min', 'E_lj_phys_median',
    'E_osc_mean',
    'E_total_mean', 'E_total_min',
    'gap_to_global_min',
    'ksd_squared', 'bandwidth',
    'rg_mean', 'rg_std',
    'nn_mean', 'nn_std',
    'coord_num_mean', 'coord_num_std',
    'frac_phys_lj_below_m140', 'frac_phys_lj_below_m160',
    'frac_phys_lj_below_m170', 'frac_phys_lj_below_m173',
    'valid_count', 'nan_count',
]


def main():
    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print("=" * 70, flush=True)
    print("LJ38 ASBS — Reference-Free Evaluation (fixed energy convention)", flush=True)
    print(f"Started: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)
    print(f"N_SAMPLES={N_SAMPLES}, N_EVAL_SEEDS={N_EVAL_SEEDS}, DEVICE={DEVICE}", flush=True)
    print(flush=True)
    print("Energy convention:", flush=True)
    print("  bgflow double-counts LJ pairs (38×37 directed, not 703 unique)", flush=True)
    print("  E_total(code) = 2×E_LJ_phys + E_osc", flush=True)
    print(f"  Known LJ38 physical min: {LJ38_PHYSICAL_GLOBAL_MIN} (FCC),", flush=True)
    print(f"                           {LJ38_PHYSICAL_ICO_MIN} (icosahedral)", flush=True)
    print("=" * 70, flush=True)

    # Load model
    exp_dir = PROJECT_ROOT / 'results' / 'lj38_asbs' / 'seed_0'
    sde, source, energy, epoch = load_model(exp_dir, 'checkpoint_latest.pt', DEVICE)

    seed_metrics = []
    for eval_seed in range(N_EVAL_SEEDS):
        print(f"\n--- Seed {eval_seed}/{N_EVAL_SEEDS - 1} ---", flush=True)
        torch.manual_seed(eval_seed * 7777)
        np.random.seed(eval_seed * 7777)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(eval_seed * 7777)

        # Generate
        t0 = time.time()
        samples = generate_samples(sde, source, N_SAMPLES, BATCH_SIZE, DEVICE)
        samples = remove_mean(samples, N_PARTICLES, SPATIAL_DIM)
        gen_time = time.time() - t0
        print(f"  Generated {N_SAMPLES} samples in {gen_time:.1f}s", flush=True)

        # Evaluate
        t0 = time.time()
        m = compute_metrics(samples, energy)
        eval_time = time.time() - t0
        m['gen_time'] = gen_time
        m['eval_time'] = eval_time
        seed_metrics.append(m)

        # Print key metrics
        print(f"  Physical E_LJ: {m.get('E_lj_phys_mean', 0):.2f} ± {m.get('E_lj_phys_std', 0):.2f} "
              f"(min={m.get('E_lj_phys_min', 0):.2f}, "
              f"gap to min={m.get('gap_to_global_min', 0):+.2f})", flush=True)
        print(f"  E_osc: {m.get('E_osc_mean', 0):.2f} ± {m.get('E_osc_std', 0):.2f}", flush=True)
        print(f"  E_total(code): {m.get('E_total_mean', 0):.2f} "
              f"(min={m.get('E_total_min', 0):.2f})", flush=True)
        print(f"  KSD²: {m.get('ksd_squared', float('nan')):.4f} "
              f"(bandwidth={m.get('bandwidth', 0):.4f})", flush=True)
        print(f"  Rg: {m.get('rg_mean', 0):.4f} ± {m.get('rg_std', 0):.4f}", flush=True)
        print(f"  Coord#: {m.get('coord_num_mean', 0):.2f} ± {m.get('coord_num_std', 0):.2f}", flush=True)
        print(f"  NN dist: {m.get('nn_mean', 0):.4f} ± {m.get('nn_std', 0):.4f}", flush=True)
        print(f"  Frac phys_LJ < -140: {m.get('frac_phys_lj_below_m140', 0):.3f}, "
              f"< -160: {m.get('frac_phys_lj_below_m160', 0):.3f}, "
              f"< -170: {m.get('frac_phys_lj_below_m170', 0):.3f}, "
              f"< -173: {m.get('frac_phys_lj_below_m173', 0):.3f}", flush=True)
        print(f"  Valid: {m.get('valid_count', 0)}/{N_SAMPLES}, "
              f"NaN: {m.get('nan_count', 0)}", flush=True)
        print(f"  Time: {gen_time:.1f}s gen + {eval_time:.1f}s eval", flush=True)

    # ---- Aggregate across seeds ----
    print(f"\n\n{'='*70}", flush=True)
    print("AGGREGATE (mean ± std across 5 seeds)", flush=True)
    print(f"{'='*70}", flush=True)

    agg = {}
    for key in SUMMARY_KEYS:
        vals = [m[key] for m in seed_metrics if key in m and np.isfinite(m.get(key, float('nan')))]
        if vals:
            agg[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}
            print(f"  {key:<30s}: {np.mean(vals):>12.4f} ± {np.std(vals):<10.4f}", flush=True)

    # ---- Save full results ----
    results = {
        'experiment': 'lj38_asbs',
        'checkpoint': 'checkpoint_latest.pt',
        'epoch': epoch,
        'n_samples': N_SAMPLES,
        'n_eval_seeds': N_EVAL_SEEDS,
        'device': DEVICE,
        'energy_convention_note': (
            'bgflow double-counts LJ pairs (38x37 directed pairs instead of 703 unique). '
            'E_total(code) = 2*E_LJ_physical + E_oscillator. '
            'Physical LJ energies (E_lj_phys_*) are the meaningful ones to compare '
            'against known LJ38 minima (-173.928 FCC, -173.252 icosahedral).'
        ),
        'known_minima_physical': {
            'global_fcc': LJ38_PHYSICAL_GLOBAL_MIN,
            'icosahedral': LJ38_PHYSICAL_ICO_MIN,
        },
        'aggregate': agg,
        'per_seed': seed_metrics,
    }

    out_path = PROJECT_ROOT / 'evaluation' / 'lj38_no_ref_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    now_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    print(f"\nResults saved to {out_path}", flush=True)
    print(f"Finished: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST", flush=True)


if __name__ == '__main__':
    main()
