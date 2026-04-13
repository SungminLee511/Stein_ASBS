"""Quick script: regenerate marginal evolution figure excluding DARW beta=0.3"""
import sys
sys.path.insert(0, '/home/sky/SML/Stein_ASBS')

import torch
import numpy as np
from pathlib import Path

from eval_grid25_darw import (
    set_neurips_style, load_model, generate_full_states,
    plot_marginal_stacked, RESULTS_DIR, FIG_DIR,
    C_ASBS, C_DARW05, C_DARW07,
)

def main():
    set_neurips_style()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_samples = 2000
    traj_seed = 42

    methods = {
        'ASBS': {
            'seed_dir': RESULTS_DIR / 'grid25_asbs' / 'seed_0',
            'color': C_ASBS,
        },
        r'DARW $\beta$=0.5': {
            'seed_dir': RESULTS_DIR / 'grid25_darw_b0.5_s0' / 'seed_0',
            'color': C_DARW05,
        },
        r'DARW $\beta$=0.7': {
            'seed_dir': RESULTS_DIR / 'grid25_darw_b0.7_s0' / 'seed_0',
            'color': C_DARW07,
        },
    }

    # Load energy
    first_dir = list(methods.values())[0]['seed_dir']
    _, _, energy, _ = load_model(first_dir, device)
    centers = energy.get_centers().to(device)

    all_states_list = []
    all_ts_list = []
    fig_method_names = []
    fig_method_colors = []

    for method_name, info in methods.items():
        print(f"  Generating trajectories for {method_name}...")
        sde, source, _, ts_cfg = load_model(info['seed_dir'], device)
        torch.manual_seed(traj_seed)
        states, ts = generate_full_states(sde, source, ts_cfg, n_samples, device)
        all_states_list.append(states)
        all_ts_list.append(ts)
        fig_method_names.append(method_name)
        fig_method_colors.append(info['color'])
        del sde, source
        torch.cuda.empty_cache()

    print("  Plotting...")
    plot_marginal_stacked(
        energy, all_states_list, all_ts_list, centers,
        fig_method_names, fig_method_colors,
        FIG_DIR / 'grid25_darw_marginal_neurips_no_b03.png',
        n_snapshots=6,
    )
    print("Done!")

if __name__ == '__main__':
    main()
