"""
select_figure2_tiles.py
=======================

Selects candidate tiles for Figure 2 (qualitative failure mode under
geographic transfer). Implements the four-stage filter chain described
in the planning document:

    Stage 1 — image-quality filters (per tile)
    Stage 2 — failure-mode filters (per tile, per cell)
    Stage 3 — median-pick within each cell
    Stage 4 — visual-quality re-rank

Inputs (per architecture x holdout-city x tile):
    - RGB input tile
    - Binary GT shadow mask
    - Upper-bound model probability map      (within-city training)
    - LOCO-vanilla model probability map     (geographic transfer)
    - SIB model probability map              (proposed method)

Outputs (in OUTPUT_DIR):
    - selection_metrics_full.csv     all tiles, all metrics, pass/fail flags
    - selection_top_candidates.csv   candidates that survive all filters
    - contact_sheet.pdf              paginated visual contact sheet
    - selection_summary.txt          plain-text top-K list with filenames

Environment variables:
    BASE_PATH     project root  (default: $PROJECT_ROOT)
    OUTPUT_DIR    output dir    (default: $BASE_PATH/data/figure2_selection)
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_PATH  = Path(os.environ.get("BASE_PATH", os.environ["PROJECT_ROOT"]))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", BASE_PATH / "data/figure2_selection"))

CITIES = ["chicago", "miami", "phoenix"]
MODELS = ["dinov3", "mamnet", "oglanet"]
RES    = "highres"

# Stage 1 — image-quality thresholds (model-independent)
COV_RANGE          = (0.08, 0.35)   # GT shadow coverage
MIN_BLOBS          = 3              # min connected shadow components
MIN_LARGEST_BLOB   = 200            # px, ensures at least one substantial shadow
LUM_RANGE          = (60, 200)      # mean luminance, avoids dark/blown tiles

# Stage 2 — failure-mode thresholds, per-architecture.
#
# DINOv3 degrades much less than the CNNs under LOCO (per Table 1, mean dmIoU
# ~ -1.7 vs -7 to -15 for MAMNet/OGLANet), so its per-tile delta_fg is smaller.
# The MAMNet/OGLANet thresholds are tuned for CNN-sized failure; DINOv3 needs
# its own floor and adds a p_upper_S floor so the visual story still reads
# (upper column must be unambiguously bright on viridis).
THRESH_BY_MODEL = {
    "dinov3": {
        "DELTA_FG_MIN":  0.06,         # smaller real collapse
        "DELTA_BG_MAX":  0.05,         # keep — narrative-critical
        "R_RANGE":       (0.40, 1.60), # wider — small Δfg blows up the ratio
        "IOU_UPPER_MIN": 0.40,
        "P_UPPER_S_MIN": 0.70,         # NEW: ensures upper column reads bright
    },
    "mamnet": {
        "DELTA_FG_MIN":  0.15,
        "DELTA_BG_MAX":  0.05,
        "R_RANGE":       (0.60, 1.30),
        "IOU_UPPER_MIN": 0.40,
        "P_UPPER_S_MIN": 0.0,          # not enforced for CNNs (already strong)
    },
    "oglanet": {
        "DELTA_FG_MIN":  0.15,
        "DELTA_BG_MAX":  0.05,
        "R_RANGE":       (0.60, 1.30),
        "IOU_UPPER_MIN": 0.40,
        "P_UPPER_S_MIN": 0.0,
    },
}

# How many candidates to retain after each stage
TOP_K_PER_CELL_S3  = 15             # closest to median delta_fg
TOP_K_PER_CELL_S4  = 8              # final per cell after quality re-rank
CONTACT_PER_ARCH   = 8              # how many to show per architecture in PDF
CANDIDATES_PER_PAGE = 4             # rows on each PDF page


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def get_paths(city: str, model: str, tile_id: str) -> dict:
    return {
        "input": BASE_PATH / f"data/Final_data_test/{city}/{RES}/test/images/{tile_id}.png",
        "gt":    BASE_PATH / f"data/Final_data_test/{city}/{RES}/test/masks/{tile_id}.png",
        "upper": BASE_PATH / f"data/Test_img_probs/upper/{city}/{RES}/{model}/base/{tile_id}.npy",
        "loco":  BASE_PATH / f"data/Test_img_probs/loco/{city}/{RES}/{model}/vanilla/{tile_id}.npy",
        "sib":   BASE_PATH / f"data/{model}/sp_gap_results/c4clean_probs_{model}_{city}_{RES}/{tile_id}.npy",
    }


def list_test_tiles(city: str) -> list:
    img_dir = BASE_PATH / f"data/Final_data_test/{city}/{RES}/test/images"
    if not img_dir.exists():
        print(f"  [warn] image directory missing: {img_dir}")
        return []
    return sorted(p.stem for p in img_dir.glob("*.png"))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask_binary(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"))
    return (m > 127).astype(np.uint8)


def load_prob_map(path: Path, target_shape) -> np.ndarray:
    """Load .npy probability/logit array and normalise to (H, W) in [0, 1]."""
    arr = np.load(path).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        # (C, H, W) or (H, W, C); pick foreground class
        if arr.shape[0] in (1, 2):
            arr = arr[1] if arr.shape[0] == 2 else arr[0]
        elif arr.shape[-1] in (1, 2):
            arr = arr[..., 1] if arr.shape[-1] == 2 else arr[..., 0]
        else:
            raise ValueError(f"Unexpected 3D shape {arr.shape} in {path}")
    if arr.ndim != 2:
        raise ValueError(f"Could not reduce to 2D: shape {arr.shape} in {path}")
    # Sigmoid if values look like logits
    if arr.min() < -1e-3 or arr.max() > 1.0 + 1e-3:
        arr = 1.0 / (1.0 + np.exp(-arr))
    arr = np.clip(arr, 0.0, 1.0)
    if arr.shape != target_shape:
        # Resize via PIL bilinear; rare but defensive
        arr = np.array(
            Image.fromarray(arr).resize((target_shape[1], target_shape[0]), Image.BILINEAR)
        )
    return arr


# ---------------------------------------------------------------------------
# Per-tile metric computation
# ---------------------------------------------------------------------------
def compute_metrics(input_img, gt, upper, loco, sib) -> dict | None:
    h, w = gt.shape
    n = h * w
    shadow = gt == 1
    bg     = gt == 0
    if shadow.sum() == 0 or bg.sum() == 0:
        return None

    m = {}
    m["cov"] = float(shadow.sum() / n)

    labels, n_blobs = ndimage.label(shadow)
    m["n_blobs"] = int(n_blobs)
    if n_blobs > 0:
        sizes = ndimage.sum(shadow, labels, range(1, n_blobs + 1))
        m["max_blob"] = float(sizes.max())
        m["min_blob"] = float(sizes.min())
    else:
        m["max_blob"] = 0.0
        m["min_blob"] = 0.0

    gray = input_img.mean(axis=-1).astype(np.float32) if input_img.ndim == 3 else input_img.astype(np.float32)
    m["mean_lum"]  = float(gray.mean())
    m["lum_std"]   = float(gray.std())
    m["lap_var"]   = float(ndimage.laplace(gray).var())

    # Foreground / background mean prob
    m["p_upper_S"] = float(upper[shadow].mean())
    m["p_loco_S"]  = float(loco[shadow].mean())
    m["p_sib_S"]   = float(sib[shadow].mean())
    m["p_upper_B"] = float(upper[bg].mean())
    m["p_loco_B"]  = float(loco[bg].mean())
    m["p_sib_B"]   = float(sib[bg].mean())

    m["delta_fg"] = m["p_upper_S"] - m["p_loco_S"]
    m["delta_bg"] = m["p_loco_B"]  - m["p_upper_B"]
    if abs(m["delta_fg"]) > 1e-4:
        m["R"] = (m["p_sib_S"] - m["p_loco_S"]) / m["delta_fg"]
    else:
        m["R"] = 0.0

    # mIoU at threshold 0.5 (mean over the two classes)
    for name, p in [("upper", upper), ("loco", loco), ("sib", sib)]:
        pred = p >= 0.5
        inter_s = float((pred & shadow).sum())
        union_s = float((pred | shadow).sum())
        iou_s = inter_s / union_s if union_s > 0 else 0.0
        inter_b = float((~pred & bg).sum())
        union_b = float((~pred | bg).sum())
        iou_b = inter_b / union_b if union_b > 0 else 0.0
        m[f"miou_{name}"]   = (iou_s + iou_b) / 2.0
        m[f"iou_s_{name}"]  = iou_s

    # Halo FP rate for LOCO: fraction of LOCO false positives that lie in a
    # 10-px boundary band around shadow regions vs total LOCO false positives.
    dilated = ndimage.binary_dilation(shadow, iterations=10)
    halo = dilated & ~shadow
    loco_pred = loco >= 0.5
    loco_fp = loco_pred & bg
    total_fp = float(loco_fp.sum())
    m["halo_fp_ratio"] = float((loco_fp & halo).sum() / total_fp) if total_fp > 0 else 0.0

    return m


def stage1_pass(m: dict) -> bool:
    return (COV_RANGE[0] <= m["cov"] <= COV_RANGE[1]
            and m["n_blobs"]   >= MIN_BLOBS
            and m["max_blob"]  >  MIN_LARGEST_BLOB
            and LUM_RANGE[0]   <= m["mean_lum"] <= LUM_RANGE[1])


def stage2_pass(m: dict, model: str) -> bool:
    t = THRESH_BY_MODEL[model]
    return (m["delta_fg"] > t["DELTA_FG_MIN"]
            and m["delta_bg"] < t["DELTA_BG_MAX"]
            and t["R_RANGE"][0] <= m["R"] <= t["R_RANGE"][1]
            and m["miou_upper"] >= t["IOU_UPPER_MIN"]
            and m["p_upper_S"] >= t["P_UPPER_S_MIN"])


# ---------------------------------------------------------------------------
# Cell ranking (Stage 3 + Stage 4)
# ---------------------------------------------------------------------------
def rank_cell(group: pd.DataFrame) -> pd.DataFrame:
    if len(group) == 0:
        return group
    median = group["delta_fg"].median()
    g = group.copy()
    g["median_distance"] = (g["delta_fg"] - median).abs()
    g = g.nsmallest(min(TOP_K_PER_CELL_S3, len(g)), "median_distance").copy()

    # z-score visual quality features within the surviving set
    def _z(s):
        sd = s.std()
        return (s - s.mean()) / (sd if sd > 1e-9 else 1.0)

    if len(g) > 1:
        g["lap_z"]   = _z(g["lap_var"])
        g["lum_z"]   = _z(g["lum_std"])
        g["halo_z"]  = _z(g["halo_fp_ratio"])
        g["quality_score"] = g["lap_z"] + g["lum_z"] - g["halo_z"]
    else:
        g["quality_score"] = 0.0

    return g.nlargest(min(TOP_K_PER_CELL_S4, len(g)), "quality_score")


# ---------------------------------------------------------------------------
# Contact sheet rendering
# ---------------------------------------------------------------------------
def render_contact_sheet(top_df: pd.DataFrame, pdf_path: Path) -> None:
    """Per-architecture pages, CONTACT_PER_ARCH rows total per architecture,
    CANDIDATES_PER_PAGE per page."""

    with PdfPages(pdf_path) as pdf:
        for model in MODELS:
            arch_rows = top_df[top_df["model"] == model] \
                            .sort_values("quality_score", ascending=False) \
                            .head(CONTACT_PER_ARCH) \
                            .reset_index(drop=True)

            if len(arch_rows) == 0:
                # Emit an empty page indicating no survivors
                fig = plt.figure(figsize=(11, 8.5))
                fig.text(0.5, 0.5,
                         f"No tiles survived all filters for {model.upper()}.\n"
                         f"Inspect selection_metrics_full.csv to relax thresholds.",
                         ha="center", va="center", fontsize=14)
                pdf.savefig(fig); plt.close(fig)
                continue

            for page_start in range(0, len(arch_rows), CANDIDATES_PER_PAGE):
                page_rows = arch_rows.iloc[page_start:page_start + CANDIDATES_PER_PAGE]
                _render_page(model, page_rows, pdf, page_start)


def _render_page(model: str, page_rows: pd.DataFrame, pdf: PdfPages, page_offset: int) -> None:
    n = len(page_rows)
    fig, axes = plt.subplots(n, 5, figsize=(11, 2.5 * n + 0.5),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.55})
    if n == 1:
        axes = axes[None, :]  # shape compat

    fig.suptitle(f"{model.upper()} candidates — page {page_offset // CANDIDATES_PER_PAGE + 1}",
                 fontsize=12, fontweight="bold", y=0.995)

    col_titles = ["Input", "GT", "Upper-bound", "LOCO (vanilla)", "SIB (ours)"]

    for i, (_, rec) in enumerate(page_rows.iterrows()):
        # Re-load images for this candidate
        paths = get_paths(rec["city"], rec["model"], rec["tile_id"])
        try:
            inp   = load_image_rgb(paths["input"])
            gt    = load_mask_binary(paths["gt"])
            upper = load_prob_map(paths["upper"], gt.shape)
            loco  = load_prob_map(paths["loco"],  gt.shape)
            sib   = load_prob_map(paths["sib"],   gt.shape)
        except Exception as e:
            for j in range(5):
                axes[i, j].axis("off")
            axes[i, 0].text(0, 0.5, f"load error: {e}", fontsize=8)
            continue

        panels = [inp, gt, upper, loco, sib]
        for j, panel in enumerate(panels):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(inp)
            elif j == 1:
                ax.imshow(gt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                ax.imshow(panel, cmap="viridis", vmin=0, vmax=1, interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10, pad=4)

        # Header for this row, above the leftmost image, full width.
        rank_in_arch = page_offset + i + 1
        header = (f"#{rank_in_arch:02d}  {rec['model'].upper()} | {rec['city']:8s}"
                  f" | {rec['tile_id']}   "
                  f"Δfg={rec['delta_fg']:.2f}  Δbg={rec['delta_bg']:.2f}  "
                  f"R={rec['R']:.2f}  "
                  f"mIoU(U/L/S)={rec['miou_upper']:.2f}/{rec['miou_loco']:.2f}/{rec['miou_sib']:.2f}  "
                  f"p̄_S(U/L/S)={rec['p_upper_S']:.2f}/{rec['p_loco_S']:.2f}/{rec['p_sib_S']:.2f}")

        # Place header just above this row's images, anchored left of column 0
        bb = axes[i, 0].get_position()
        fig.text(bb.x0, bb.y1 + 0.012, header,
                 fontsize=8, family="monospace",
                 ha="left", va="bottom")

    plt.subplots_adjust(top=0.92, bottom=0.02, left=0.02, right=0.98)
    pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plain-text summary
# ---------------------------------------------------------------------------
def write_summary(top_df: pd.DataFrame, summary_path: Path) -> None:
    lines = []
    lines.append("Figure 2 — top candidates per architecture")
    lines.append("=" * 78)
    lines.append("")
    for model in MODELS:
        sub = top_df[top_df["model"] == model] \
                 .sort_values("quality_score", ascending=False) \
                 .head(CONTACT_PER_ARCH)
        lines.append(f"[{model.upper()}]   ({len(sub)} candidates)")
        lines.append("-" * 78)
        if len(sub) == 0:
            lines.append("  (no tiles survived all filters)")
            lines.append("")
            continue
        header = f"{'#':>2}  {'city':8s}  {'tile_id':40s}  {'Δfg':>5}  {'Δbg':>5}  {'R':>5}  {'mIoU_L':>6}"
        lines.append(header)
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            lines.append(f"{i:>2}  {r['city']:8s}  {r['tile_id']:40s}  "
                         f"{r['delta_fg']:>5.2f}  {r['delta_bg']:>5.2f}  "
                         f"{r['R']:>5.2f}  {r['miou_loco']:>6.3f}")
        lines.append("")
    summary_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"BASE_PATH  = {BASE_PATH}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print()

    records = []
    skip_counts = {"missing_file": 0, "load_error": 0, "degenerate_mask": 0}

    for model in MODELS:
        for city in CITIES:
            tiles = list_test_tiles(city)
            print(f"[{model:8s} | {city:8s}]   {len(tiles)} tiles", flush=True)
            for tid in tiles:
                paths = get_paths(city, model, tid)

                # Existence check first — a missing file is the most common case
                missing = [k for k, v in paths.items() if not v.exists()]
                if missing:
                    skip_counts["missing_file"] += 1
                    continue

                try:
                    inp   = load_image_rgb(paths["input"])
                    gt    = load_mask_binary(paths["gt"])
                    upper = load_prob_map(paths["upper"], gt.shape)
                    loco  = load_prob_map(paths["loco"],  gt.shape)
                    sib   = load_prob_map(paths["sib"],   gt.shape)
                except Exception as e:
                    skip_counts["load_error"] += 1
                    print(f"    [skip] {tid}: {e}")
                    continue

                m = compute_metrics(inp, gt, upper, loco, sib)
                if m is None:
                    skip_counts["degenerate_mask"] += 1
                    continue

                m["model"]        = model
                m["city"]         = city
                m["tile_id"]      = tid
                m["stage1_pass"]  = stage1_pass(m)
                m["stage2_pass"]  = stage2_pass(m, model)
                records.append(m)

    print()
    print(f"Skipped: {skip_counts}")
    if not records:
        print("No tiles processed. Aborting.")
        sys.exit(1)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "selection_metrics_full.csv", index=False)
    print(f"Wrote selection_metrics_full.csv  ({len(df)} rows)")

    # Filter pipeline
    surviving = df[df["stage1_pass"] & df["stage2_pass"]].copy()
    print(f"Stage 1 pass: {df['stage1_pass'].sum():>5} / {len(df)}")
    print(f"Stage 2 pass: {df['stage2_pass'].sum():>5} / {len(df)}")
    print(f"Both stages : {len(surviving):>5} / {len(df)}")

    if len(surviving) == 0:
        print("No tiles survived Stage 1 + 2. Inspect CSV and relax thresholds.")
        sys.exit(1)

    ranked = surviving.groupby(["model", "city"], group_keys=False).apply(rank_cell)
    ranked.to_csv(OUTPUT_DIR / "selection_top_candidates.csv", index=False)
    print(f"Wrote selection_top_candidates.csv  ({len(ranked)} rows)")

    # Contact sheet + summary
    pdf_path = OUTPUT_DIR / "contact_sheet.pdf"
    summary_path = OUTPUT_DIR / "selection_summary.txt"
    render_contact_sheet(ranked, pdf_path)
    write_summary(ranked, summary_path)
    print(f"Wrote contact_sheet.pdf")
    print(f"Wrote selection_summary.txt")
    print()
    print("Done.")


if __name__ == "__main__":
    main()