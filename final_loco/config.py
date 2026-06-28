"""
Configuration for cross-location diagnostic analysis.
All paths, constants, and mappings defined here.

Server: uncomment the block for your active server; NCSA Delta is active.
"""
import os

# ============================================================
# PATHS — uncomment the block for your server
# ============================================================

# All paths resolve from the single PROJECT_ROOT environment variable
# (see .env.example / top-level README). Per-cluster differences live there.
_ROOT = os.environ["PROJECT_ROOT"]

# --- Gilbreth ---
# GT_BASE      = os.path.join(_ROOT, "data", "Final_data_test")
# PRED_BASE    = os.path.join(_ROOT, "data", "Test_img_results")
# OUTPUT_BASE  = os.path.join(_ROOT, "data", "loco_diagnostic_results")
# FEATURE_BASE = os.path.join(_ROOT, "data", "extracted_features")

# --- NCSA Delta ---
GT_BASE      = os.path.join(_ROOT, "data", "Final_data_test")
PRED_BASE    = os.path.join(_ROOT, "data", "Test_img_results")
OUTPUT_BASE  = os.path.join(_ROOT, "data", "loco_diagnostic_results")
FEATURE_BASE = os.path.join(_ROOT, "data", "extracted_features")

# ============================================================
# EVALUATION CONSTANTS
# ============================================================
IMG_SIZE          = 384
BOUNDARY_WIDTH    = 2    # pixels — tolerant ±k zone half-width
MIN_INSTANCE_AREA = 50   # minimum pixels for a valid shadow instance

# ============================================================
# DOMAIN CONFIGURATION
# ============================================================
CITIES      = ["chicago", "miami", "phoenix"]
RESOLUTIONS = ["highres", "midres"]
MODELS      = ["mamnet", "oglanet", "dinov3"]

# MCL removed.  New: iim, isw (same base arch — training-only reg.),
# mrfp_plus (training-only perturbation), fada (adapters active at inference).
LOCO_VARIANTS = ["vanilla", "fda", "segdesic", "iim", "isw", "mrfp_plus", "fada"]

# ============================================================
# SHADOW TYPE MAPPING
# ============================================================
SHADOW_TYPE_MAP = {
    0: "Background",
    1: "Building/canyon",
    2: "Under-structure",
    3: "Tree-canopy",
    4: "Topography-cast",
    5: "Vehicle-cast",
    6: "Thin-linear",
}

SHADOW_TYPE_SHORT = {
    1: "Building",
    2: "Under-struct",
    3: "Tree",
    4: "Topo",
    5: "Vehicle",
    6: "Thin-linear",
}

# ============================================================
# PATH HELPERS
# ============================================================

def gt_mask_dir(city, res):
    return os.path.join(GT_BASE, city, res, "test", "masks")

def gt_multiclass_dir(city, res):
    return os.path.join(GT_BASE, city, res, "test", "masks_multiclass")

def image_dir(city, res):
    return os.path.join(GT_BASE, city, res, "test", "images")

def pred_dir(test_type, city, res, model, variant):
    return os.path.join(PRED_BASE, test_type, city, res, model, variant)

def upper_pred_dir(city, res, model):
    return pred_dir("upper", city, res, model, "base")

def loco_pred_dir(city, res, model, variant):
    return pred_dir("loco", city, res, model, variant)

def output_dir(thread, subdir=""):
    p = os.path.join(OUTPUT_BASE, thread, subdir)
    os.makedirs(p, exist_ok=True)
    return p