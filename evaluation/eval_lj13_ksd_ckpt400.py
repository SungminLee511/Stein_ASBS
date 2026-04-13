"""Quick eval for LJ13 KSD-ASBS checkpoint_400 only."""
import sys, os, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, numpy as np
from pathlib import Path

# Force flush on every print
import functools
print = functools.partial(print, flush=True)

from eval_lj13 import load_model, generate_samples, compute_metrics, PROJECT_ROOT, N_EVAL_SEEDS
from adjoint_samplers.utils.graph_utils import remove_mean

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 2000

def main():
    try:
        print('=== LJ13 KSD-ASBS Checkpoint 400 Evaluation ===')
        print(f'Device: {DEVICE}')

        ref_path = PROJECT_ROOT / 'data' / 'test_split_LJ13-1000.npy'
        ref_samples = np.load(ref_path, allow_pickle=True)
        print(f'Reference samples: {ref_samples.shape}')
        ref_torch = torch.tensor(ref_samples, dtype=torch.float32).to(DEVICE)
        ref_torch = remove_mean(ref_torch, 13, 3)

        ksd_dir = PROJECT_ROOT / 'results' / 'lj13_ksd_asbs' / 'seed_0'
        config_path = ksd_dir / 'config.yaml'
        ckpt_path = ksd_dir / 'checkpoints' / 'checkpoint_400.pt'
        print(f'Checkpoint: {ckpt_path}')

        sde, source, energy, ts_cfg, cfg = load_model(config_path, ckpt_path, DEVICE)
        print('Model loaded successfully.')

        seed_results = []
        for seed in range(N_EVAL_SEEDS):
            print(f'\n--- Seed {seed}/{N_EVAL_SEEDS} ---')
            torch.manual_seed(seed * 7777)
            np.random.seed(seed * 7777)

            print('  Generating samples...')
            samples = generate_samples(sde, source, ts_cfg, N_SAMPLES, DEVICE)
            samples = remove_mean(samples, 13, 3)
            print('  Computing metrics...')
            m = compute_metrics(samples, energy, ref_torch)
            seed_results.append(m)

            for k in ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared', 'mean_energy', 'valid_count']:
                if k in m:
                    print(f'    {k}: {m[k]}')

        print(f'\n{"="*60}')
        print('AGGREGATED RESULTS (mean +/- std over 5 seeds)')
        print(f'{"="*60}')
        numeric_keys = ['energy_w2', 'eq_w2', 'dist_w2', 'ksd_squared', 'mean_energy',
                        'std_energy', 'min_energy', 'max_energy', 'valid_count', 'nan_count']
        for key in numeric_keys:
            vals = [m[key] for m in seed_results if key in m and not (isinstance(m[key], float) and np.isnan(m[key]))]
            if vals:
                print(f'  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}')
        print('Done!')

    except Exception as e:
        print(f'\n!!! EVAL FAILED !!!')
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
