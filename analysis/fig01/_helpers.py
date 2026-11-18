#!/usr/bin/env python3
"""
Figure 1: ShadowTransfer — v3 (paired test sampling + redesigned layout)
========================================================================
Changes from v2:
  - Tiles are paired by filename in the test/ split: same stem in highres/test/
    and midres/test/ → same coordinate at both resolutions.
  - Wide-short footprint (14 × 6 in, ~2.33:1).
  - 3×2 grid: columns = Chicago/Miami/Phoenix, rows = 0.3 m / 0.6 m.
  - No top banner, no TRAIN/TEST badges, no chart insets.
  - Drop chart with numeric mIoU axis, Δ labels, and an in-domain ↔ LOCO
    reference fan that makes the architecture-dependent pattern visible at
    thumbnail size.
  - Single full-width shadow-category legend strip below.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
from scipy.ndimage import binary_dilation

warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = os.path.join(os.environ["PROJECT_ROOT"], "data/Final_data_test")
CSV_PATH   = os.path.join(os.environ["PROJECT_ROOT"], "data/model_results/statistical_results.csv")
OUTPUT_DIR = os.path.join(os.environ["PROJECT_ROOT"], "data/final_figs")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "figure1_shadowtransfer.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "figure1_shadowtransfer.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── COLOURS ───────────────────────────────────────────────────────────────────
WHITE    = '#FFFFFF'
TEXT     = '#1F2937'
TEXT_DIM = '#6B7280'
BORDER   = '#E5E7EB'
GRID_COL = '#ECECF2'

# Six-class shadow palette (kept; subtler edges)
CLASS_FILL = {
    1: (0.15, 0.50, 0.95, 0.50),  # building / canyon
    2: (0.95, 0.32, 0.12, 0.50),  # under-structure
    3: (0.10, 0.75, 0.30, 0.50),  # tree-canopy dapple
    4: (0.72, 0.12, 0.88, 0.50),  # topography-cast
    5: (0.95, 0.78, 0.05, 0.50),  # vehicle-cast
    6: (0.05, 0.85, 0.88, 0.50),  # thin linear
}
CLASS_EDGE = {k: (*v[:3], 0.92) for k, v in CLASS_FILL.items()}
CLASS_NAME = {
    1: 'Building / canyon', 2: 'Under-structure',
    3: 'Tree-canopy dapple', 4: 'Topography-cast',
    5: 'Vehicle-cast',       6: 'Thin linear',
}

# Sophisticated, print-safe city accents (slate / terracotta / clay)
CITY_COL   = {
    'chicago': '#3B5C7A',
    'miami':   '#C97A4F',
    'phoenix': '#8B4A3C',
}
CITY_LABEL = {'chicago': 'Chicago', 'miami': 'Miami', 'phoenix': 'Phoenix'}

# Architecture accents (steel / forest / bronze)
MODEL_COL = {
    'dinov3':  '#1F6FAA',
    'mamnet':  '#3F7A4F',
    'oglanet': '#A6622E',
}
MODEL_LABEL = {
    'dinov3':  'DINOv3',
    'mamnet':  'MAMNet',
    'oglanet': 'OGLANet',
}


# ── PAIRED IMAGE LOADING ──────────────────────────────────────────────────────

def shadow_frac_mc(mc_path):
    try:
        m = np.array(Image.open(str(mc_path)))
        return float((m > 0).mean())
    except Exception:
        return 0.0


def count_classes_mc(mc_path):
    try:
        m = np.array(Image.open(str(mc_path)))
        return int(len(np.unique(m[m > 0])))
    except Exception:
        return 0


def _pair_key(stem):
    """Strip the resolution token so highres and midres patches share a key.

    e.g. 'miami_session20_highres_paired_004'
       → 'miami_session20_paired_004'
         'miami_session20_midres_paired_004'  also → 'miami_session20_paired_004'
    """
    return stem.replace('_highres_', '_').replace('_midres_', '_')


def find_paired_test_stems(city):
    """Return list of (pair_key, hr_stem, mr_stem) for paired test patches."""
    base = Path(DATA_ROOT) / city
    hr_dir = base / 'highres' / 'test' / 'images'
    mr_dir = base / 'midres'  / 'test' / 'images'
    if not hr_dir.exists() or not mr_dir.exists():
        return []
    hr_map = {_pair_key(p.stem): p.stem for p in hr_dir.glob('*.png')}
    mr_map = {_pair_key(p.stem): p.stem for p in mr_dir.glob('*.png')}
    common = sorted(set(hr_map) & set(mr_map))
    return [(k, hr_map[k], mr_map[k]) for k in common]


def score_pair(city, hr_stem, mr_stem):
    """Score a paired (hr, mr) test patch by combined class diversity and
    shadow coverage. Both resolutions must have masks and non-trivial shadow.
    """
    base = Path(DATA_ROOT) / city
    hr_mc = base / 'highres' / 'test' / 'masks_multiclass' / f'{hr_stem}.png'
    mr_mc = base / 'midres'  / 'test' / 'masks_multiclass' / f'{mr_stem}.png'
    if not (hr_mc.exists() and mr_mc.exists()):
        return None

    hr_classes = count_classes_mc(hr_mc)
    mr_classes = count_classes_mc(mr_mc)
    hr_frac    = shadow_frac_mc(hr_mc)
    mr_frac    = shadow_frac_mc(mr_mc)
    if min(hr_frac, mr_frac) < 0.05:
        return None

    return {
        'hr_stem': hr_stem, 'mr_stem': mr_stem,
        'hr_classes': hr_classes, 'mr_classes': mr_classes,
        'hr_frac':    hr_frac,    'mr_frac':    mr_frac,
        'classes_min': min(hr_classes, mr_classes),
        'classes_sum': hr_classes + mr_classes,
        'frac_min':    min(hr_frac, mr_frac),
        'frac_sum':    hr_frac + mr_frac,
    }


def load_paired_test(city, prefer_idx=0):
    """Best paired test patch for a city. The two stems differ in their
    resolution token but refer to the same coordinate.
    """
    pairs  = find_paired_test_stems(city)
    scored = [s for s in (score_pair(city, hr, mr) for _, hr, mr in pairs)
              if s is not None]

    if not scored:
        print(f"  [WARN] {city}: no paired test patches — using placeholder")
        hr_img, hr_mc = _placeholder(city, 'highres')
        mr_img, mr_mc = _placeholder(city, 'midres')
        return (hr_img, hr_mc), (mr_img, mr_mc), None

    # Rank: more classes (visual richness), then more shadow coverage
    scored.sort(key=lambda s: (-s['classes_min'], -s['classes_sum'],
                               -s['frac_min'],   -s['frac_sum']))
    pick = scored[min(prefer_idx, len(scored) - 1)]

    base = Path(DATA_ROOT) / city
    hr_img = np.array(Image.open(
        base/'highres'/'test'/'images'/f"{pick['hr_stem']}.png").convert('RGB'))
    hr_mc  = np.array(Image.open(
        base/'highres'/'test'/'masks_multiclass'/f"{pick['hr_stem']}.png"))
    mr_img = np.array(Image.open(
        base/'midres' /'test'/'images'/f"{pick['mr_stem']}.png").convert('RGB'))
    mr_mc  = np.array(Image.open(
        base/'midres' /'test'/'masks_multiclass'/f"{pick['mr_stem']}.png"))

    print(f"  {city}: hr={pick['hr_stem']}  mr={pick['mr_stem']}  "
          f"(classes hr/mr={pick['hr_classes']}/{pick['mr_classes']}, "
          f"frac hr/mr={pick['hr_frac']:.1%}/{pick['mr_frac']:.1%})")
    return (hr_img, hr_mc), (mr_img, mr_mc), pick['hr_stem']


def _placeholder(city, resolution):
    rng  = np.random.default_rng(abs(hash(city + resolution)) % 2**31)
    base = {'chicago':[120,130,150],'miami':[145,165,115],'phoenix':[185,165,125]}[city]
    img  = np.full((384, 384, 3), base, dtype=np.uint8)
    for _ in range(25):
        y0, x0 = rng.integers(0, 350, 2); h, w = rng.integers(10, 55, 2)
        img[y0:y0+h, x0:x0+w] = rng.integers(90, 230, 3, dtype=np.uint8)
    mc = np.zeros((384, 384), np.uint8)
    for k in [1, 2, 3, 4, 5]:
        for _ in range(5):
            cy, cx = rng.integers(40, 340, 2)
            ry, rx = rng.integers(18, 70), rng.integers(25, 100)
            ang    = rng.uniform(0, np.pi)
            yy, xx = np.ogrid[:384, :384]
            yr =  (yy - cy) * np.cos(ang) + (xx - cx) * np.sin(ang)
            xr = -(yy - cy) * np.sin(ang) + (xx - cx) * np.cos(ang)
            mc[((yr / ry) ** 2 + (xr / rx) ** 2) < 1] = k
    if resolution == 'midres':
        img = (img.astype(float) * 0.87 + 18).clip(0, 255).astype(np.uint8)
    return img, mc


def overlay_multiclass(img, mc_mask):
    rgba = np.zeros((*img.shape[:2], 4), np.float32)
    rgba[..., :3] = img / 255.0
    rgba[..., 3]  = 1.0
    for k, (r, g, b, a) in CLASS_FILL.items():
        px = mc_mask == k
        if not px.any():
            continue
        rgba[px, 0] = rgba[px, 0] * (1 - a) + r * a
        rgba[px, 1] = rgba[px, 1] * (1 - a) + g * a
        rgba[px, 2] = rgba[px, 2] * (1 - a) + b * a
        er, eg, eb, ea = CLASS_EDGE[k]
        edge = binary_dilation(px, iterations=1) & ~px
        rgba[edge] = [er, eg, eb, ea]
    return rgba


