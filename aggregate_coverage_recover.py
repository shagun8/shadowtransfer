"""
aggregate_coverage_recover.py — Phase 2 coverage-recovery test.

For each of 9 cells (3 archs × 3 cities):
  1. Read per-cell SP-gap JSON (must contain val_rc_records + test_rc_records).
  2. Fit c*_val on aggregate source-val RC curve using fixed β.
  3. Compute per-image test selective error at c*_val and at c=1.0.
  4. Image-level cluster bootstrap on per-image deltas (B=10000).
Population test:
  Wilcoxon signed-rank on n=9 cell-mean deltas, one-sided H1: median < 0.
"""
import os, sys, json, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from scipy.stats import wilcoxon
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

ARCHITECTURES = ['mamnet', 'oglanet', 'dinov3']
CITIES        = ['phoenix', 'miami', 'chicago']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mamnet_results_dir',  required=True)
    p.add_argument('--oglanet_results_dir', required=True)
    p.add_argument('--dinov3_results_dir',  required=True)
    p.add_argument('--output_dir',          required=True)
    p.add_argument('--resolution',          default='highres')
    p.add_argument('--bootstrap_B',  type=int,   default=10000)
    p.add_argument('--beta',         type=float, default=0.5,
                   help='Source-val error reduction factor for c*_val')
    p.add_argument('--fallback_c',   type=float, default=1.0,
                   help='If no c satisfies beta-criterion, use this')
    return p.parse_args()


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


def aggregate_rc(records):
    """Mean RC curve over images. Skips images with None entries (too few shadow px)."""
    curves = []
    for r in records:
        rc = r['rc_curve']
        if rc is None or any(x is None for x in rc):
            continue
        curves.append(rc)
    if not curves:
        return None
    return np.mean(np.array(curves, dtype=np.float64), axis=0)


def per_image_err_at_c(records, c, coverage_grid):
    """Per-image selective error at coverage c. Returns dict {filename: err} (NaN-skipped)."""
    grid = np.asarray(coverage_grid)
    idx  = int(np.argmin(np.abs(grid - c)))
    out  = {}
    for r in records:
        rc = r['rc_curve']
        if rc is None or rc[idx] is None:
            continue
        stem = os.path.splitext(r['filename'])[0]
        out[stem] = float(rc[idx])
    return out


def bootstrap_delta_ci(deltas, B=10000, seed=42):
    valid = deltas[~np.isnan(deltas)]
    n = len(valid)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), 0
    obs = float(np.mean(valid))
    rng = np.random.RandomState(seed)
    boot = np.array([float(np.mean(valid[rng.choice(n, n, replace=True)]))
                     for _ in range(B)])
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    if obs < 0:
        p = 2.0 * max(float(np.mean(boot >= 0)), 1.0/B)
    else:
        p = 2.0 * max(float(np.mean(boot <= 0)), 1.0/B)
    return obs, lo, hi, min(p, 1.0), n


def find_cell_json(results_dir, arch, city, resolution):
    for fname in [f'sp_gap_{arch}_{city}_{resolution}.json',
                  f'sp_gap_{arch}_c4clean_{city}_{resolution}.json']:
        p = Path(results_dir) / fname
        if p.exists():
            return p
    return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    arch_dirs = {'mamnet':  args.mamnet_results_dir,
                 'oglanet': args.oglanet_results_dir,
                 'dinov3':  args.dinov3_results_dir}

    cell_results = {}
    cell_aurc_deltas = []   # for population Wilcoxon (one per cell)
    cell_labels      = []

    print(f'\n{"="*70}')
    print(f'PHASE 2: coverage recovery, beta={args.beta}, fallback_c={args.fallback_c}')
    print(f'{"="*70}')

    for arch in ARCHITECTURES:
        for city in CITIES:
            jpath = find_cell_json(arch_dirs[arch], arch, city, args.resolution)
            if jpath is None:
                print(f'  {arch}/{city}: JSON not found, skipping')
                continue
            with open(jpath) as f:
                data = json.load(f)
            if 'val_rc_records' not in data or 'test_rc_records' not in data:
                print(f'  {arch}/{city}: missing val/test_rc_records '
                      f'(rerun sp_gap_analysis with the patch); skipping')
                continue

            grid       = np.asarray(data['coverage_grid'])
            val_agg    = aggregate_rc(data['val_rc_records'])
            test_recs  = data['test_rc_records']
            if val_agg is None:
                print(f'  {arch}/{city}: empty val_rc; skipping')
                continue

            c_star, val_full, val_target = find_c_star_val(
                val_agg, grid, args.beta, args.fallback_c)

            # per-image test selective error at c*_val and at c=1.0
            err_at_c_star = per_image_err_at_c(test_recs, c_star, grid)
            err_at_c_1    = per_image_err_at_c(test_recs, 1.0,    grid)

            common = sorted(set(err_at_c_star) & set(err_at_c_1))
            if not common:
                print(f'  {arch}/{city}: no paired test images; skipping')
                continue

            deltas = np.array([err_at_c_star[s] - err_at_c_1[s] for s in common],
                              dtype=np.float64)
            mean_d, lo, hi, p, n_v = bootstrap_delta_ci(deltas, B=args.bootstrap_B)
            n_neg = int(np.sum(deltas < 0))

            cell_results[f'{arch}/{city}'] = {
                'arch': arch, 'city': city,
                'c_star_val':       c_star,
                'val_full_err':     val_full,
                'val_target':       val_target,
                'val_full_at_c_star':  float(val_agg[int(np.argmin(np.abs(grid - c_star)))]),
                'test_full_err':    float(np.mean(list(err_at_c_1.values()))),
                'test_err_at_c_star': float(np.mean(list(err_at_c_star.values()))),
                'mean_delta':       mean_d,
                'ci_lo':            lo,
                'ci_hi':            hi,
                'p_two_sided':      p,
                'n_paired':         n_v,
                'n_improve':        n_neg,
            }
            cell_aurc_deltas.append(mean_d)
            cell_labels.append(f'{arch}/{city}')

            print(f'  {arch:8s}/{city:8s}: '
                  f'c*={c_star:.2f}  '
                  f'test_err {np.mean(list(err_at_c_1.values())):.4f}'
                  f' -> {np.mean(list(err_at_c_star.values())):.4f}  '
                  f'Δ={mean_d:+.4f} [{lo:+.4f}, {hi:+.4f}]  '
                  f'{n_neg}/{n_v} improve')

    # ---- Population Wilcoxon ----
    print(f'\n{"="*70}')
    print(f'POPULATION TEST (n={len(cell_aurc_deltas)} cells)')
    print(f'{"="*70}')

    pop = {'n_cells': len(cell_aurc_deltas),
           'cell_aurc_deltas': cell_aurc_deltas,
           'cell_labels':      cell_labels,
           'mean_across_cells': float(np.mean(cell_aurc_deltas))
                                if cell_aurc_deltas else float('nan')}
    if SCIPY_OK and len(cell_aurc_deltas) >= 6:
        try:
            stat_two, p_two   = wilcoxon(cell_aurc_deltas, alternative='two-sided',
                                          zero_method='wilcox')
            stat_less, p_less = wilcoxon(cell_aurc_deltas, alternative='less',
                                          zero_method='wilcox')
            pop['wilcoxon_two_sided_p'] = float(p_two)
            pop['wilcoxon_p_less']      = float(p_less)
            pop['n_negative'] = int(np.sum(np.array(cell_aurc_deltas) < 0))
            pop['n_zero']     = int(np.sum(np.array(cell_aurc_deltas) == 0))
            print(f'  mean ΔAURC across cells = {pop["mean_across_cells"]:+.4f}')
            print(f'  Wilcoxon (H1: C4 < c=1)  p_less   = {p_less:.5f}')
            print(f'  Wilcoxon (two-sided)     p        = {p_two:.5f}')
            print(f'  cells improving          {pop["n_negative"]}/{pop["n_cells"]}')
        except ValueError as e:
            pop['note'] = f'Wilcoxon failed: {e}'
            print(f'  Wilcoxon failed: {e}')
    else:
        pop['note'] = 'scipy missing or n<6 cells'
        print(f'  Wilcoxon skipped: {pop["note"]}')

    # ---- Save ----
    final = {
        'config': {
            'resolution':  args.resolution,
            'bootstrap_B': args.bootstrap_B,
            'beta':        args.beta,
            'fallback_c':  args.fallback_c,
        },
        'cell_results':      cell_results,
        'population':        pop,
    }
    out = os.path.join(args.output_dir, 'coverage_recover_results.json')
    with open(out, 'w') as f:
        json.dump(final, f, indent=2, default=str)
    print(f'\nSaved → {out}')

    # ---- Markdown ----
    md = [f'# Phase 2 Coverage Recovery — β={args.beta}', '',
          f'**Cells:** {len(cell_aurc_deltas)} / 9  '
          f'**Mean ΔAURC across cells:** '
          f'{pop["mean_across_cells"]:+.4f}']
    if 'wilcoxon_p_less' in pop:
        md.append(f'**Wilcoxon p (H1: C4 reduces error):** {pop["wilcoxon_p_less"]:.5f}')
        md.append(f'**Cells improving:** {pop["n_negative"]}/{pop["n_cells"]}')
    md += ['', '## Per-cell',
           '| Arch | City | c*_val | Test err @ c=1 | Test err @ c* | Δ | 95% CI | n improve | p |',
           '|---|---|---|---|---|---|---|---|---|']
    for k, v in cell_results.items():
        md.append(f'| {v["arch"]} | {v["city"]} | {v["c_star_val"]:.2f} | '
                  f'{v["test_full_err"]:.4f} | {v["test_err_at_c_star"]:.4f} | '
                  f'{v["mean_delta"]:+.4f} | '
                  f'[{v["ci_lo"]:+.4f}, {v["ci_hi"]:+.4f}] | '
                  f'{v["n_improve"]}/{v["n_paired"]} | {v["p_two_sided"]:.4f} |')
    with open(os.path.join(args.output_dir, 'coverage_recover_summary.md'), 'w') as f:
        f.write('\n'.join(md))
    print(f'Saved → {os.path.join(args.output_dir, "coverage_recover_summary.md")}')


if __name__ == '__main__':
    main()