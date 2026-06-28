#!/usr/bin/env python3
"""
Phase 1 aggregator: read 9 cell_result.json files, run the pre-registered
decision rule on AURC_shadow.

Decision rule:
  PASS  : Wilcoxon p<0.05 (one-sided) AND ≥7/9 cells with ΔAURC < 0
  PASS-STRAT: 6/9 improve, both CNNs all 3/3, DINOv3 null/slight, CNN mean Δ < -0.03
  FAIL  : <6/9 improve OR Wilcoxon p > 0.20  → Phase 2 takes over
"""

import os, sys, json, argparse
from datetime import datetime
import numpy as np
from scipy.stats import wilcoxon

ARCHITECTURES = ['mamnet', 'oglanet', 'dinov3']
CITIES = ['phoenix', 'miami', 'chicago']

PRE_REG_PATH = 'pre_registration.json'

PRE_REG = {
    'primary_endpoint': 'AURC_shadow',
    'alternative':      'C4-clean reduces AURC_shadow vs Vanilla',
    'test':             'one-sided Wilcoxon signed-rank, alternative="less"',
    'alpha':            0.05,
    'n_cells':          9,
    'fixed_coverage_c': 0.6,
    'rationale':        ('AURC_shadow is the §4.3 headline metric; one-sided '
                          'Wilcoxon on 9 cell-mean deltas; min achievable p ≈ 0.002. '
                          'Coverage c=0.6 derived from §4.3 Phoenix median.'),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--phase1_root', required=True,
                   help='Root containing the 9 <arch>__holdout_<city>/ subdirs')
    p.add_argument('--output_dir', required=True)
    return p.parse_args()


def load_cells(root):
    cells = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry, 'cell_result.json')
        if os.path.isfile(path):
            with open(path) as f:
                cells.append(json.load(f))
    return cells


def write_pre_registration(output_dir):
    """Write pre-registration once; refuse to overwrite."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, PRE_REG_PATH)
    if os.path.isfile(path):
        with open(path) as f:
            existing = json.load(f)
        if existing.get('primary_endpoint') != PRE_REG['primary_endpoint']:
            raise RuntimeError(
                f'Pre-registration already exists at {path} with different '
                f'primary endpoint. Refusing to overwrite.')
        print(f'  Pre-registration locked: {path}')
        return existing
    PRE_REG['locked_at'] = datetime.now().isoformat()
    with open(path, 'w') as f:
        json.dump(PRE_REG, f, indent=2)
    print(f'  Pre-registration written: {path}')
    return PRE_REG


def main():
    args = parse_args()
    pre_reg = write_pre_registration(args.output_dir)

    cells = load_cells(args.phase1_root)
    if len(cells) < 9:
        print(f'  Found {len(cells)}/9 cells. Continuing with what is available.')

    # ─── Per-cell deltas ───
    per_arch = {a: [] for a in ARCHITECTURES}
    all_aurc = []
    all_ece = []
    all_miou = []

    print('\n' + '=' * 90)
    print(f'  PHASE 1 RESULTS — {len(cells)} cells loaded')
    print('=' * 90)
    print(f'  {"Arch":<10} {"Holdout":<10} {"ΔAURC":>10} {"95%CI":>20} '
          f'{"ΔECE_pos":>10} {"ΔmIoU":>10}')
    print('  ' + '-' * 86)

    for cell in cells:
        arch = cell['arch']
        holdout = cell['holdout']
        a = cell['phase1']['aurc_shadow']
        e = cell['phase1']['ece_pred_pos']
        m = cell['phase1']['miou']
        ci_str = f'[{a["ci_lo"]:+.4f},{a["ci_hi"]:+.4f}]'
        print(f'  {arch:<10} {holdout:<10} {a["delta_mean"]:+10.5f} '
              f'{ci_str:>20} {e["delta_mean"]:+10.5f} {m["delta_mean"]:+10.5f}')
        per_arch[arch].append((holdout, a['delta_mean'], a['ci_lo'], a['ci_hi'],
                                e['delta_mean'], m['delta_mean']))
        all_aurc.append(a['delta_mean'])
        all_ece.append(e['delta_mean'])
        all_miou.append(m['delta_mean'])

    # ─── Population-level test ───
    print('\n' + '=' * 90)
    print('  PRIMARY ENDPOINT: ΔAURC_shadow (one-sided Wilcoxon, alternative="less")')
    print('=' * 90)
    aurc_arr = np.array(all_aurc)
    n_negative = int((aurc_arr < 0).sum())
    n_total = len(aurc_arr)

    if n_total >= 6:
        # one-sided: H1 says median delta < 0
        try:
            stat, p_one = wilcoxon(aurc_arr, alternative='less',
                                    zero_method='wilcox')
        except ValueError as e:
            stat, p_one = float('nan'), float('nan')
            print(f'  Wilcoxon failed: {e}')
    else:
        stat, p_one = float('nan'), float('nan')

    print(f'  Cells with ΔAURC < 0: {n_negative}/{n_total}')
    print(f'  Wilcoxon statistic:   {stat}')
    print(f'  One-sided p-value:    {p_one:.4f}')
    print(f'  Mean ΔAURC across cells: {np.mean(aurc_arr):+.5f}')
    print(f'  Median ΔAURC across cells: {np.median(aurc_arr):+.5f}')

    # ─── Decision rule ───
    print('\n' + '=' * 90)
    print('  DECISION RULE')
    print('=' * 90)

    pass_full = (p_one < pre_reg['alpha']) and (n_negative >= 7)

    # Stratified pass: CNNs improve uniformly, DINOv3 may not
    cnn_archs = ['mamnet', 'oglanet']
    cnn_neg = sum(1 for a in cnn_archs
                   for (_, d, *_) in per_arch[a]
                   if d < 0)
    cnn_total = sum(len(per_arch[a]) for a in cnn_archs)
    cnn_mean = np.mean([d for a in cnn_archs
                        for (_, d, *_) in per_arch[a]])
    dino_deltas = [d for (_, d, *_) in per_arch['dinov3']]
    pass_strat = (
        n_negative >= 6
        and cnn_neg == cnn_total          # both CNN archs all negative
        and cnn_mean < -0.03
    )

    decision = 'FAIL'
    if pass_full:
        decision = 'PASS'
    elif pass_strat:
        decision = 'PASS_STRATIFIED'
    elif n_negative < 6 or (not np.isnan(p_one) and p_one > 0.20):
        decision = 'FAIL_GO_PHASE2'

    print(f'  Decision: {decision}')

    if decision == 'PASS':
        print('  → §5 narrative: C4-clean reduces SP-gap as a representational')
        print('    side-effect of §4.2+§4.4 modules. No coverage thresholding needed.')
    elif decision == 'PASS_STRATIFIED':
        print('  → §5 narrative: SP-gap recovers on CNN architectures (per §4.4 locus);')
        print(f'    DINOv3 shows no further recovery (CNN mean Δ={cnn_mean:+.4f},')
        print(f'    DINOv3 deltas={[round(d, 4) for d in dino_deltas]}).')
    else:
        print('  → Move to Phase 2: fixed coverage c=0.6 from §4.3 population.')

    # ─── Phase 2 (only printed if Phase 1 failed) ───
    if decision == 'FAIL_GO_PHASE2':
        print('\n' + '=' * 90)
        print('  PHASE 2: SELECTIVE ERROR @ c=0.6 vs c=1.0  (C4-clean)')
        print('=' * 90)
        sel_deltas = []
        for cell in cells:
            sel = cell['phase2']['selective_error_c06_minus_c10']
            sel_deltas.append(sel['delta_mean'])
            print(f'  {cell["arch"]:<10} {cell["holdout"]:<10} '
                  f'Δ={sel["delta_mean"]:+.5f}  '
                  f'CI=[{sel["ci_lo"]:+.5f},{sel["ci_hi"]:+.5f}]')
        sel_arr = np.array(sel_deltas)
        n_neg_p2 = int((sel_arr < 0).sum())
        try:
            _, p_p2 = wilcoxon(sel_arr, alternative='less')
        except ValueError:
            p_p2 = float('nan')
        print(f'\n  Cells with Δ<0: {n_neg_p2}/{len(sel_arr)}')
        print(f'  Wilcoxon one-sided p: {p_p2:.4f}')
        print(f'  Mean Δ: {np.mean(sel_arr):+.5f}')

    # ─── Save aggregated report ───
    report = {
        'pre_registration': pre_reg,
        'n_cells_loaded':   len(cells),
        'per_cell':         [{
            'arch': c['arch'],
            'holdout': c['holdout'],
            'phase1': c['phase1'],
            'phase2': c['phase2'],
        } for c in cells],
        'phase1_population': {
            'mean_delta_aurc':   float(np.mean(aurc_arr)),
            'median_delta_aurc': float(np.median(aurc_arr)),
            'n_negative':        n_negative,
            'wilcoxon_p_oneside': float(p_one) if not np.isnan(p_one) else None,
        },
        'per_arch_summary': {
            arch: {
                'n':          len(per_arch[arch]),
                'mean_aurc':  float(np.mean([d for (_, d, *_) in per_arch[arch]])
                                    if per_arch[arch] else float('nan')),
                'n_negative': sum(1 for (_, d, *_) in per_arch[arch] if d < 0),
            } for arch in ARCHITECTURES
        },
        'decision': decision,
    }
    if decision == 'FAIL_GO_PHASE2':
        report['phase2_population'] = {
            'mean_delta_sel_err':   float(np.mean(sel_deltas)),
            'n_negative':           n_neg_p2,
            'wilcoxon_p_oneside':   float(p_p2) if not np.isnan(p_p2) else None,
        }

    out_path = os.path.join(args.output_dir, 'phase1_report.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n  Full report → {out_path}')


if __name__ == '__main__':
    main()