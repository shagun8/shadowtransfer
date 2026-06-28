"""
verify_c_star_val.py — visual sanity check on the c*_val operating point.

For each of 9 cells, plots the source-val aggregate selective-shadow-error
risk-coverage curve, with the chosen c*_val and the beta-criterion target
overlaid. Confirms that c*_val sits at a genuine "halve the source-val
full-coverage error" operating point and is not a discretization artifact
of the coverage grid.

Reads the augmented sp_gap_*_c4clean_*.json files (must contain
val_rc_records and coverage_grid; produced by sp_gap_analysis*.py after
the val-inference patch).

Output: one summary grid PNG (3x3, archs as rows, cities as columns) in
the verification dir.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ARCHITECTURES = ['mamnet', 'oglanet', 'dinov3']
CITIES        = ['phoenix', 'miami', 'chicago']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mamnet_results_dir',  required=True)
    p.add_argument('--oglanet_results_dir', required=True)
    p.add_argument('--dinov3_results_dir',  required=True)
    p.add_argument('--output_dir',          required=True)
    p.add_argument('--resolution',          default='highres')
    p.add_argument('--beta',     type=float, default=0.5)
    p.add_argument('--fallback_c', type=float, default=1.0)
    return p.parse_args()


def find_cell_json(results_dir, arch, city, resolution):
    for fname in [f'sp_gap_{arch}_{city}_{resolution}.json',
                  f'sp_gap_{arch}_c4clean_{city}_{resolution}.json']:
        p = Path(results_dir) / fname
        if p.exists():
            return p
    return None


def aggregate_rc(records):
    """Mean RC curve over images. Skips images with None entries."""
    curves = []
    for r in records:
        rc = r['rc_curve']
        if rc is None or any(x is None for x in rc):
            continue
        curves.append(rc)
    if not curves:
        return None
    return np.mean(np.array(curves, dtype=np.float64), axis=0)


def find_c_star_val(val_rc_aggregate, coverage_grid, beta, fallback=1.0):
    """Largest c such that aggregate val sel-err(c) <= sel-err(c=1.0)*(1-beta)."""
    val_full = float(val_rc_aggregate[-1])
    if val_full <= 0:
        return fallback, val_full, val_full * (1 - beta)
    target = val_full * (1.0 - beta)
    valid = val_rc_aggregate <= target
    if not np.any(valid):
        return fallback, val_full, target
    return float(coverage_grid[np.where(valid)[0].max()]), val_full, target


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    arch_dirs = {'mamnet':  args.mamnet_results_dir,
                 'oglanet': args.oglanet_results_dir,
                 'dinov3':  args.dinov3_results_dir}

    print(f'\n{"="*70}')
    print(f'c*_val verification — beta={args.beta}, resolution={args.resolution}')
    print(f'{"="*70}')

    # 3x3 grid: rows = architectures, cols = cities
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=120, sharey=False)

    summary_rows = []

    for r, arch in enumerate(ARCHITECTURES):
        for c, city in enumerate(CITIES):
            ax = axes[r, c]
            jpath = find_cell_json(arch_dirs[arch], arch, city, args.resolution)
            if jpath is None:
                ax.set_title(f'{arch}/{city}\n(JSON not found)', fontsize=10)
                ax.axis('off')
                continue
            with open(jpath) as f:
                data = json.load(f)
            if 'val_rc_records' not in data or 'coverage_grid' not in data:
                ax.set_title(f'{arch}/{city}\n(no val_rc_records)', fontsize=10)
                ax.axis('off')
                continue

            grid    = np.asarray(data['coverage_grid'])
            val_agg = aggregate_rc(data['val_rc_records'])
            if val_agg is None:
                ax.set_title(f'{arch}/{city}\n(empty val_rc)', fontsize=10)
                ax.axis('off')
                continue

            c_star, val_full, val_target = find_c_star_val(
                val_agg, grid, args.beta, args.fallback_c)

            val_at_cstar = float(val_agg[int(np.argmin(np.abs(grid - c_star)))])
            achieved_reduction = (val_full - val_at_cstar) / val_full \
                if val_full > 0 else float('nan')

            # ---- Plot ----
            ax.plot(grid, val_agg, 'o-', color='#1f77b4', linewidth=1.6,
                    markersize=4, label='Source-val sel-err curve')
            # Target line: horizontal at val_target
            ax.axhline(val_target, color='#d62728', linestyle='--',
                       linewidth=1.2, alpha=0.8,
                       label=f'Target = (1-β)·err(1) = {val_target:.4f}')
            # c*_val vertical line
            ax.axvline(c_star, color='#2ca02c', linestyle=':',
                       linewidth=1.6,
                       label=f'c*_val = {c_star:.2f}')
            # Highlight (c*_val, sel-err at c*_val)
            ax.plot([c_star], [val_at_cstar], '*', color='#2ca02c',
                    markersize=14, markeredgecolor='black',
                    markeredgewidth=0.5)

            ax.set_xlabel('Coverage c', fontsize=9)
            ax.set_ylabel('Source-val selective shadow error', fontsize=9)
            ax.set_xlim(0.05, 1.05)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            ax.legend(loc='upper left', fontsize=7)

            title_color = 'green' if achieved_reduction >= args.beta - 0.02 else 'orange'
            ax.set_title(
                f'{arch} / {city}\n'
                f'val full-err = {val_full:.4f}  |  '
                f'val err@c* = {val_at_cstar:.4f}  ({achieved_reduction*100:.0f}% reduction)',
                fontsize=10, color=title_color)

            print(f'  {arch:8s}/{city:8s}: c*={c_star:.2f}  '
                  f'val_full={val_full:.4f}  val@c*={val_at_cstar:.4f}  '
                  f'reduction={achieved_reduction*100:.1f}%  '
                  f'target={val_target:.4f}')

            summary_rows.append({
                'arch':            arch,
                'city':            city,
                'c_star_val':      c_star,
                'val_full_err':    val_full,
                'val_err_at_cstar': val_at_cstar,
                'val_target':      val_target,
                'achieved_reduction_frac': achieved_reduction,
                'beta_criterion_met': achieved_reduction >= args.beta - 0.02,
            })

    fig.suptitle(
        f'c*_val operating-point verification (β={args.beta})\n'
        f'Green star = chosen c*_val on source-val RC curve. '
        f'Target = source-val full-coverage error × (1−β).',
        fontsize=12, y=1.00)
    plt.tight_layout()

    grid_path = os.path.join(
        args.output_dir,
        f'verify_c_star_val_grid_beta{args.beta}.png')
    plt.savefig(grid_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'\nSaved grid plot → {grid_path}')

    # ---- Markdown summary ----
    md = [f'# c*_val verification — β={args.beta}', '',
          f'**Resolution:** {args.resolution}  '
          f'**Cells:** {len(summary_rows)} / 9',
          '',
          '| Arch | City | c*_val | Source-val full err | Source-val err @ c* | Reduction | β criterion met |',
          '|---|---|---|---|---|---|---|']
    for s in summary_rows:
        check = '✓' if s['beta_criterion_met'] else '⚠'
        md.append(
            f'| {s["arch"]} | {s["city"]} | '
            f'{s["c_star_val"]:.2f} | '
            f'{s["val_full_err"]:.4f} | '
            f'{s["val_err_at_cstar"]:.4f} | '
            f'{s["achieved_reduction_frac"]*100:.1f}% | '
            f'{check} |'
        )
    md.append('')
    md.append('**Interpretation.** A green star resting on or below the dashed '
              'red line indicates c*_val achieves the β-criterion ((1−β) '
              'reduction in source-val selective error vs full coverage). '
              'A green-titled cell passed the criterion; an orange-titled '
              'cell achieved less than the targeted reduction (typically '
              'because the source-val RC curve flattens before the criterion '
              'is met — c*_val falls back to the largest grid value still '
              'below target, which may slightly under-shoot β due to coverage '
              'grid discretization).')
    md.append('')
    md.append('A `c*_val` value of 1.00 indicates the fallback was triggered '
              '(no coverage on the grid satisfied the criterion); if any cell '
              'shows c*_val=1.00 here, the corresponding test-side delta will '
              'be 0 by construction. The aggregator output flags such cells.')
    md_path = os.path.join(
        args.output_dir,
        f'verify_c_star_val_summary_beta{args.beta}.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))
    print(f'Saved summary  → {md_path}')


if __name__ == '__main__':
    main()