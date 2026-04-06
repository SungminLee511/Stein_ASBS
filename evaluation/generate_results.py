"""
generate_results.py

Reads results/ directory and generates RESULTS.md with:
- Per-benchmark comparison tables (baseline vs KSD-ASBS)
- lambda ablation tables and plots
- Batch size ablation
- Chunking timing table
- Synthetic rotated GMM mode coverage
- Energy histogram overlays
- Dimension scaling plots

Usage:
    python generate_results.py --results_dir results --output RESULTS.md
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_all_results(results_dir):
    """Load all JSON result files."""
    data = {}
    for f in Path(results_dir).glob('*_results.json'):
        group = f.stem.replace('_results', '')
        with open(f) as fh:
            data[group] = json.load(fh)
    # Chunking timing
    timing_path = Path(results_dir) / 'chunking_timing.json'
    if timing_path.exists():
        with open(timing_path) as f:
            data['_chunking'] = json.load(f)
    return data


def fmt(val, key=''):
    """Format a metric value."""
    if isinstance(val, dict):
        m, s = val.get('mean', 0), val.get('std', 0)
        if abs(m) < 0.01:
            return f"{m:.2e} +/- {s:.2e}"
        return f"{m:.4f} +/- {s:.4f}"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def make_benchmark_table(group_data, benchmark, metrics, fig_dir):
    """Generate markdown table + energy histogram for one benchmark."""
    lines = []
    lines.append(f"### {benchmark.upper()}")
    lines.append("")

    # Find baseline and KSD experiments
    baseline_exps = {k: v for k, v in group_data.items() if 'ksd' not in k}
    ksd_exps = {k: v for k, v in group_data.items() if 'ksd' in k}

    if not baseline_exps and not ksd_exps:
        lines.append("_No data available._")
        return "\n".join(lines)

    # Aggregate across seeds
    def aggregate_seeds(exps):
        agg = defaultdict(list)
        for exp_name, exp_data in exps.items():
            for metric in metrics:
                if metric in exp_data and isinstance(exp_data[metric], dict):
                    agg[metric].extend(exp_data[metric].get('values', []))
        result = {}
        for metric, vals in agg.items():
            result[metric] = {'mean': np.mean(vals), 'std': np.std(vals)}
        return result

    base_agg = aggregate_seeds(baseline_exps) if baseline_exps else {}
    ksd_agg = aggregate_seeds(ksd_exps) if ksd_exps else {}

    # Table
    lines.append("| Metric | Baseline ASBS | KSD-ASBS | Delta (%) |")
    lines.append("|---|---|---|---|")
    for metric in metrics:
        base_str = fmt(base_agg[metric]) if metric in base_agg else "—"
        ksd_str = fmt(ksd_agg[metric]) if metric in ksd_agg else "—"
        if metric in base_agg and metric in ksd_agg:
            bm = base_agg[metric]['mean']
            km = ksd_agg[metric]['mean']
            delta = ((km - bm) / (abs(bm) + 1e-10)) * 100
            better = "✓" if (km < bm and 'coverage' not in metric) or \
                            (km > bm and 'coverage' in metric) else ""
            delta_str = f"{delta:+.1f}% {better}"
        else:
            delta_str = "—"
        lines.append(f"| {metric} | {base_str} | {ksd_str} | {delta_str} |")
    lines.append("")

    # Energy histogram
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        for exp_name, exp_data in baseline_exps.items():
            if '_energy_values' in exp_data:
                E = exp_data['_energy_values']
                if isinstance(E, list) and len(E) > 0:
                    ax.hist(E, bins=50, alpha=0.5, density=True,
                            label='Baseline ASBS', color='#1f77b4')
                break
        for exp_name, exp_data in ksd_exps.items():
            if '_energy_values' in exp_data:
                E = exp_data['_energy_values']
                if isinstance(E, list) and len(E) > 0:
                    ax.hist(E, bins=50, alpha=0.5, density=True,
                            label='KSD-ASBS', color='#ff7f0e')
                break
        ax.set_xlabel('Energy')
        ax.set_ylabel('Density')
        ax.set_title(f'{benchmark.upper()}: Energy Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig_path = fig_dir / f'{benchmark}_energy_hist.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        lines.append(f"![{benchmark} energy histogram](results/figures/{benchmark}_energy_hist.png)")
        lines.append("")
    except Exception as e:
        lines.append(f"_Could not generate energy histogram: {e}_")
        lines.append("")

    return "\n".join(lines)


def make_lambda_ablation(group_data, benchmark, fig_dir):
    """Generate lambda ablation table and plot."""
    lines = []
    lines.append(f"### {benchmark.upper()} — Lambda Ablation")
    lines.append("")

    # Parse lambda from experiment names
    lambda_data = {}
    for exp_name, exp_data in group_data.items():
        if 'ksd_l' in exp_name:
            parts = exp_name.split('_l')
            if len(parts) > 1:
                lam_str = parts[1].split('_')[0]
                try:
                    lam = float(lam_str)
                    if lam not in lambda_data:
                        lambda_data[lam] = defaultdict(list)
                    for metric, val in exp_data.items():
                        if isinstance(val, dict) and 'mean' in val:
                            lambda_data[lam][metric].append(val['mean'])
                except ValueError:
                    pass

    if not lambda_data:
        lines.append("_No lambda ablation data found._")
        return "\n".join(lines)

    # Table
    lambdas = sorted(lambda_data.keys())
    key_metrics = ['energy_w2', 'ksd_squared', 'mean_energy']
    available = [m for m in key_metrics if m in lambda_data[lambdas[0]]]

    header = "| lambda | " + " | ".join(available) + " |"
    sep = "|---|" + "|".join(["---"] * len(available)) + "|"
    lines.append(header)
    lines.append(sep)
    for lam in lambdas:
        row = f"| {lam} |"
        for metric in available:
            vals = lambda_data[lam][metric]
            m, s = np.mean(vals), np.std(vals)
            row += f" {m:.4f} +/- {s:.4f} |"
        lines.append(row)
    lines.append("")

    # Plot
    try:
        fig, axes = plt.subplots(1, max(len(available), 1),
                                  figsize=(5 * max(len(available), 1), 4))
        if len(available) == 1:
            axes = [axes]
        for ax, metric in zip(axes, available):
            means = [np.mean(lambda_data[l][metric]) for l in lambdas]
            stds = [np.std(lambda_data[l][metric]) for l in lambdas]
            ax.errorbar(lambdas, means, yerr=stds, marker='o', capsize=3, linewidth=2)
            ax.set_xlabel('lambda (KSD weight)')
            ax.set_ylabel(metric)
            ax.set_xscale('log')
            ax.grid(alpha=0.3)
        fig.suptitle(f'{benchmark.upper()}: Lambda Ablation')
        fig.tight_layout()
        fig_path = fig_dir / f'{benchmark}_lambda_ablation.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        lines.append(f"![{benchmark} lambda ablation](results/figures/{benchmark}_lambda_ablation.png)")
        lines.append("")
    except Exception as e:
        lines.append(f"_Could not generate lambda ablation plot: {e}_")
        lines.append("")

    return "\n".join(lines)


def make_mode_coverage_table(all_data, fig_dir):
    """Generate mode coverage comparison for rotated GMM experiments."""
    lines = []
    lines.append("## Synthetic Benchmark: Rotated Gaussian Mixture (CV-Unknown)")
    lines.append("")
    lines.append("These experiments test mode coverage on energy functions where")
    lines.append("collective variables are unknown by construction (the modes are")
    lines.append("separated along a randomly rotated axis).")
    lines.append("")

    dims = []
    for key in sorted(all_data.keys()):
        if key.startswith('rotgmm'):
            dims.append(key)

    if not dims:
        lines.append("_No rotated GMM experiments found._")
        return "\n".join(lines)

    # Collect coverage data
    coverage_data = {'baseline': {}, 'ksd': {}}
    for dim_key in dims:
        dim_num = dim_key.replace('rotgmm', '')
        group = all_data[dim_key]
        for exp_name, exp_data in group.items():
            if 'coverage_fraction' in exp_data:
                cf = exp_data['coverage_fraction']
                val = cf['mean'] if isinstance(cf, dict) else cf
                if 'ksd' in exp_name:
                    coverage_data['ksd'][dim_num] = val
                else:
                    coverage_data['baseline'][dim_num] = val

    # Table
    lines.append("| Dimension | Baseline Coverage | KSD Coverage | Delta |")
    lines.append("|---|---|---|---|")
    dim_nums = sorted(set(list(coverage_data['baseline'].keys()) +
                          list(coverage_data['ksd'].keys())),
                      key=lambda x: int(x))
    for d in dim_nums:
        bc = coverage_data['baseline'].get(d, float('nan'))
        kc = coverage_data['ksd'].get(d, float('nan'))
        delta = kc - bc if not (np.isnan(bc) or np.isnan(kc)) else float('nan')
        bc_str = f"{bc:.2%}" if not np.isnan(bc) else "—"
        kc_str = f"{kc:.2%}" if not np.isnan(kc) else "—"
        delta_str = f"{delta:+.2%}" if not np.isnan(delta) else "—"
        lines.append(f"| d={d} | {bc_str} | {kc_str} | {delta_str} |")
    lines.append("")

    # Plot
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        x_dims = [int(d) for d in dim_nums]
        bc_vals = [coverage_data['baseline'].get(d, 0) for d in dim_nums]
        kc_vals = [coverage_data['ksd'].get(d, 0) for d in dim_nums]
        w = 0.35
        x = np.arange(len(x_dims))
        ax.bar(x - w/2, bc_vals, w, label='Baseline ASBS', color='#1f77b4')
        ax.bar(x + w/2, kc_vals, w, label='KSD-ASBS', color='#ff7f0e')
        ax.set_xticks(x)
        ax.set_xticklabels([f'd={d}' for d in x_dims])
        ax.set_ylabel('Mode Coverage Fraction')
        ax.set_title('Mode Coverage vs Dimension (Rotated GMM)')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig_path = fig_dir / 'rotgmm_coverage.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        lines.append(f"![Mode coverage](results/figures/rotgmm_coverage.png)")
        lines.append("")
    except Exception as e:
        lines.append(f"_Could not generate mode coverage plot: {e}_")
        lines.append("")

    return "\n".join(lines)


def make_chunking_table(chunking_data):
    """Generate chunking timing comparison table."""
    lines = []
    lines.append("## Chunking Analysis")
    lines.append("")
    lines.append("Wall-clock time for computing the Stein kernel gradient sum")
    lines.append("(N=512 particles). Chunking is mathematically equivalent")
    lines.append("(max absolute difference shown).")
    lines.append("")
    lines.append("| Dimension | Full (s) | Chunk-128 (s) | Chunk-256 (s) | Slowdown | Max Diff |")
    lines.append("|---|---|---|---|---|---|")

    for dim_label, t in chunking_data.items():
        slowdown = t['chunk128_time'] / (t['full_time'] + 1e-10)
        lines.append(
            f"| {dim_label} | {t['full_time']:.4f} | {t['chunk128_time']:.4f} | "
            f"{t['chunk256_time']:.4f} | {slowdown:.1f}x | {t['max_diff_chunk128']:.2e} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_results_md(args):
    """Generate the complete RESULTS.md."""
    results_dir = Path(args.results_dir)
    fig_dir = results_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_all_results(results_dir)

    lines = []
    lines.append("# Results: KSD-Augmented ASBS — Comprehensive Evaluation")
    lines.append("")
    lines.append("*Auto-generated by `generate_results.py`*")
    lines.append("")

    # --- Summary ---
    lines.append("## Method")
    lines.append("")
    lines.append("KSD-Augmented ASBS modifies the adjoint terminal condition:")
    lines.append("")
    lines.append("$$Y_1^i = -\\frac{1}{N}\\nabla\\Phi_0(X_1^i) - "
                 "\\frac{\\lambda}{N^2}\\sum_j \\nabla_x k_p(X_1^i, X_1^j)$$")
    lines.append("")

    # --- Per-benchmark results ---
    particle_metrics = ['energy_w2', 'dist_w2', 'eq_w2', 'ksd_squared', 'mean_energy']
    simple_metrics = ['energy_w2', 'ksd_squared', 'mean_energy']

    lines.append("## Molecular Benchmarks")
    lines.append("")
    for benchmark in ['dw4', 'lj13', 'lj55']:
        if benchmark in all_data:
            lines.append(make_benchmark_table(
                all_data[benchmark], benchmark, particle_metrics, fig_dir
            ))

    # --- Non-molecular benchmarks ---
    lines.append("## Non-Molecular Benchmarks")
    lines.append("")
    for benchmark in ['muller', 'blogreg']:
        if benchmark in all_data:
            lines.append(make_benchmark_table(
                all_data[benchmark], benchmark, simple_metrics, fig_dir
            ))

    # --- Lambda ablation ---
    lines.append("## Lambda Ablation")
    lines.append("")
    for benchmark in ['dw4', 'lj13']:
        if benchmark in all_data:
            lines.append(make_lambda_ablation(all_data[benchmark], benchmark, fig_dir))

    # --- Synthetic experiments ---
    lines.append(make_mode_coverage_table(all_data, fig_dir))

    # --- Chunking ---
    if '_chunking' in all_data:
        lines.append(make_chunking_table(all_data['_chunking']))

    # --- Conclusions ---
    lines.append("## Conclusions")
    lines.append("")
    lines.append("*(To be written based on experimental results)*")
    lines.append("")
    lines.append("Key questions answered:")
    lines.append("")
    lines.append("1. Does KSD-ASBS reduce mode collapse compared to baseline ASBS?")
    lines.append("2. What is the optimal lambda?")
    lines.append("3. Does the advantage persist in high dimensions?")
    lines.append("4. Does KSD-ASBS work where CVs are unknown (rotated GMM)?")
    lines.append("5. What is the computational overhead?")
    lines.append("6. Is chunking mathematically equivalent and practically efficient?")

    # --- Reproduction ---
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("bash scripts/run_phase2_baselines.sh  # Train baselines")
    lines.append("bash scripts/run_phase3_ksd.sh        # Train KSD-ASBS")
    lines.append("bash scripts/run_phase4_synthetic.sh   # Train on rotated GMM")
    lines.append("bash scripts/run_phase5_evaluate.sh    # Evaluate everything")
    lines.append("python generate_results.py             # Generate this report")
    lines.append("```")

    # Write
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write("\n".join(lines))
    print(f"RESULTS.md written to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate RESULTS.md from evaluation data')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--output', default='RESULTS.md')
    args = parser.parse_args()
    generate_results_md(args)
