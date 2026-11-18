#!/usr/bin/env python3
"""Figure 1 (teaser), rebuilt for the final paper.

Copied from Shade Maps/Python/FINAL-fig1.ipynb (cells 1 and 4); the original is untouched.
Changes: chart values read from results JSONs (two-class boundary-tolerant mIoU at K=2,
Vanilla, high-resolution cells, 3 held-out cities); CrossEarth added (brown, diamond marker,
plots/colors.md); architecture colours from plots/colors.md; 95% CI error bars from the CI
results JSON; the LOCO rotation glyph drawn as an inset instead of pasted on afterwards.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["PROJECT_ROOT"])   # results/ and plots/ live under PROJECT_ROOT
exec(open(HERE / '_helpers.py').read())          # loaders, tile overlay, class + city palettes
CITIES_ORDER = ['chicago', 'miami', 'phoenix']
CITY_SHORT = {'chicago': 'Chi', 'miami': 'Mia', 'phoenix': 'Phx'}
HOLDOUT_PER_ROW = [2, 1, 0]

R35 = ROOT / 'results' / '35_table2_final.json'
CI_JSON = Path(sys.argv[1]) if len(sys.argv) > 1 else None
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'plots' / '19_teaser.png'

MODELS = ['dinov3', 'mamnet', 'oglanet', 'crossearth']
ARCH_COL = {'dinov3': '#664c94', 'mamnet': '#259a9b', 'oglanet': '#eeb76b', 'crossearth': '#8c6d31'}
ARCH_MARK = {'dinov3': 'o', 'mamnet': 'o', 'oglanet': 'o', 'crossearth': 'D'}
ARCH_LABEL = {'dinov3': 'DINOv3', 'mamnet': 'MAMNet', 'oglanet': 'OGLANet', 'crossearth': 'CrossEarth'}

def chart_values():
    d = json.load(open(R35))['figure1_data']['per_arch']
    out = {m: {'upper': d[m]['high_res_only']['upper_mean'], 'loco': d[m]['high_res_only']['loco_mean'],
               'delta': d[m]['high_res_only']['delta_mean'], 'n_cells': d[m]['high_res_only']['n_cells']} for m in MODELS}
    if CI_JSON and CI_JSON.exists():
        ci = json.load(open(CI_JSON))['per_arch']
        for m in MODELS:
            for k in ('upper', 'loco', 'delta'):
                out[m][k + '_ci'] = tuple(ci[m][k]['ci95'])
                assert abs(ci[m][k]['mean'] - out[m][k]) < 1e-6, (m, k)   # CI file must describe the same values
    return out

def draw_glyph(ax):
    ax.set_xlim(0, 5.0); ax.set_ylim(0.4, 4.7); ax.set_aspect('equal'); ax.axis('off')
    ax.text(2.5, 4.40, 'LOCO  ·  3-way rotation', ha='center', va='center', fontsize=10, fontweight='bold', color=TEXT)
    col_x = [1.7, 2.7, 3.7]
    for cx, key in zip(col_x, CITIES_ORDER):
        ax.text(cx, 3.65, CITY_SHORT[key], ha='center', va='center', fontsize=9, fontweight='bold', color=CITY_COL[key])
    ax.plot([1.3, 4.1], [3.32, 3.32], color='#D8D8E0', lw=0.7)
    for y, hold in zip([2.95, 2.35, 1.75], HOLDOUT_PER_ROW):
        for ci_, (cx, key) in enumerate(zip(col_x, CITIES_ORDER)):
            c = CITY_COL[key]
            if ci_ == hold:
                ax.add_patch(mpatches.Circle((cx, y), 0.20, facecolor='white', edgecolor=c, linewidth=2.0, zorder=3))
            else:
                ax.add_patch(mpatches.Circle((cx, y), 0.20, facecolor=c, edgecolor='white', linewidth=1.1, zorder=3))
    ax.add_patch(mpatches.Circle((1.40, 0.95), 0.14, facecolor=TEXT_DIM, edgecolor='white', linewidth=0.9))
    ax.text(1.65, 0.95, 'train', ha='left', va='center', fontsize=8.5, color=TEXT)
    ax.add_patch(mpatches.Circle((2.85, 0.95), 0.14, facecolor='white', edgecolor=TEXT_DIM, linewidth=1.6))
    ax.text(3.10, 0.95, 'held out', ha='left', va='center', fontsize=8.5, color=TEXT)

def build(vals, preview):
    fig = plt.figure(figsize=(14.0, 6.0), dpi=300); fig.patch.set_facecolor(WHITE)
    outer = gridspec.GridSpec(2, 1, figure=fig, top=0.93, bottom=0.04, left=0.045, right=0.985, hspace=0.22, height_ratios=[14, 1.0])
    main = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.13, width_ratios=[55, 45])
    grid = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=main[0], hspace=0.10, wspace=0.06)
    leftmost = {}
    for ci_, city in enumerate(CITIES_ORDER):
        (hr_img, hr_mc), (mr_img, mr_mc), stem = load_paired_test(city, prefer_idx=0)
        for ri, (img, mc, res_label) in enumerate([(hr_img, hr_mc, '0.3 m/px'), (mr_img, mr_mc, '0.6 m/px')]):
            ax = fig.add_subplot(grid[ri, ci_]); comp = overlay_multiclass(img, mc)
            ax.imshow(comp, interpolation='lanczos', aspect='equal'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor(CITY_COL[city]); sp.set_linewidth(1.6)
            if ri == 0: ax.set_title(CITY_LABEL[city], fontsize=15, fontweight='bold', color=CITY_COL[city], pad=7)
            ax.text(0.97, 0.04, res_label, transform=ax.transAxes, fontsize=9.5, fontweight='bold', color='white', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.22', facecolor='#000000B0', edgecolor='none'))
            if ri == 1:
                H, W = comp.shape[:2]; bh, bw = H // 2, W // 2; y0, x0 = (H - bh) // 2, (W - bw) // 2
                ax.add_patch(mpatches.Rectangle((x0, y0), bw, bh, lw=3.6, edgecolor='black', facecolor='none', alpha=0.40, zorder=4))
                ax.add_patch(mpatches.Rectangle((x0, y0), bw, bh, lw=1.8, edgecolor='white', facecolor='none', zorder=5))
            if ci_ == 0: leftmost[ri] = ax
    for ri, label in enumerate(['0.3 m/px', '0.6 m/px']):
        leftmost[ri].annotate(label, xy=(0, 0.5), xycoords='axes fraction', xytext=(-14, 0), textcoords='offset points',
                              rotation=90, fontsize=10.5, fontweight='bold', color=TEXT_DIM, ha='center', va='center')

    ax_c = fig.add_subplot(main[1]); ax_c.set_facecolor(WHITE)
    for s in ['top', 'right']: ax_c.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax_c.spines[s].set_edgecolor(BORDER); ax_c.spines[s].set_linewidth(1.0)
    xs = range(len(MODELS))
    ax_c.plot(xs, [vals[m]['upper'] for m in MODELS], color='#BFBFCC', lw=1.4, ls='--', zorder=1, alpha=0.75)
    ax_c.plot(xs, [vals[m]['loco'] for m in MODELS], color='#888899', lw=1.4, ls=':', zorder=1, alpha=0.75)
    allv = []
    for i, m in enumerate(MODELS):
        up, lo, col, mk = vals[m]['upper'], vals[m]['loco'], ARCH_COL[m], ARCH_MARK[m]
        ax_c.plot([i, i], [up, lo], color=col, lw=3.4, alpha=0.45, zorder=2, solid_capstyle='round')
        for key, face, edge, z in (('upper', 'white', col, 4), ('loco', col, 'white', 5)):
            v = vals[m][key]; allv.append(v)
            if key + '_ci' in vals[m]:
                lo_ci, hi_ci = vals[m][key + '_ci']; allv += [lo_ci, hi_ci]
                ax_c.errorbar([i], [v], yerr=[[v - lo_ci], [hi_ci - v]], fmt='none', ecolor='#333333', elinewidth=1.1, capsize=4, zorder=3)
            ax_c.scatter([i], [v], s=170 if mk == 'o' else 140, marker=mk, facecolor=face, edgecolor=edge, linewidths=2.2, zorder=z)
        txt = f"Δ = {vals[m]['delta']:+.1f}".replace('-', '\u2212')
        if 'delta_ci' in vals[m]: txt += f"\n[{vals[m]['delta_ci'][0]:+.1f}, {vals[m]['delta_ci'][1]:+.1f}]".replace('-', '\u2212')
        ax_c.annotate(txt, xy=(i, (up + lo) / 2.0), xytext=(13, 0), textcoords='offset points', ha='left', va='center',
                      fontsize=10.5, fontweight='bold', color=col if m != 'oglanet' else '#b07d2e')
    span = max(max(allv) - min(allv), 5.0)
    ax_c.set_xlim(-0.55, len(MODELS) - 0.25); ax_c.set_ylim(min(allv) - span * 0.25, max(allv) + span * 0.75)
    ax_c.set_xticks(list(xs)); ax_c.set_xticklabels([ARCH_LABEL[m] for m in MODELS], fontsize=12, fontweight='bold', color=TEXT)
    ax_c.tick_params(axis='x', length=0, pad=8); ax_c.tick_params(axis='y', colors=TEXT_DIM, labelsize=10)
    ax_c.yaxis.grid(True, color=GRID_COL, lw=0.8, zorder=0); ax_c.set_axisbelow(True)
    ax_c.set_ylabel('mIoU (%, two-class, ±2 px tolerance)', fontsize=11, color=TEXT, labelpad=7)
    h_up = plt.Line2D([], [], marker='o', ls='None', markerfacecolor='white', markeredgecolor='#444', markeredgewidth=2.0, markersize=9, label='Within-city upper bound')
    h_lo = plt.Line2D([], [], marker='o', ls='None', markerfacecolor='#444', markeredgecolor='white', markeredgewidth=1.6, markersize=9, label='LOCO (held-out city)')
    h_ci = plt.Line2D([], [], color='#333333', lw=1.1, marker='_', markersize=8, label='95% CI, n = 450 test images')
    ax_c.legend(handles=[h_up, h_lo, h_ci], fontsize=9.5, framealpha=0.97, facecolor=WHITE, edgecolor=BORDER, labelcolor=TEXT,
                loc='upper left', handlelength=1.2, borderpad=0.6)
    gx = ax_c.inset_axes([0.60, 0.70, 0.40, 0.30]); draw_glyph(gx)
    if preview:
        ax_c.text(0.5, 0.5, 'PREVIEW: 95% CIs pending', transform=ax_c.transAxes, ha='center', va='center', fontsize=16, color='red', alpha=0.5, rotation=20)
    ax_l = fig.add_subplot(outer[1]); ax_l.axis('off')
    patches = [mpatches.Patch(facecolor=CLASS_FILL[k][:3], alpha=0.82, edgecolor=CLASS_EDGE[k][:3], lw=1.0, label=CLASS_NAME[k]) for k in sorted(CLASS_FILL)]
    ax_l.legend(handles=patches, loc='center', ncol=6, frameon=False, fontsize=10.5, labelcolor=TEXT, handlelength=1.5, handletextpad=0.55, columnspacing=1.8)
    return fig

if __name__ == '__main__':
    vals = chart_values(); preview = not all('delta_ci' in vals[m] for m in MODELS)
    for m in MODELS: print(m, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in vals[m].items()})
    fig = build(vals, preview); OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white'); print('saved', OUT, 'preview' if preview else 'final')
