#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: newmod_ablation_analysis_submit.sh
#
# Diagnostic-Module Ablation Study — N0..N12  (CACR + CE-AURC + TENT)
#
# Experiment matrix:
# ┌──────┬──────────────────────────────────────────┬──────────────────────────────┐
# │ ID   │ What's added on top of C4 (SIB-Full)     │ Question being answered      │
# ├──────┼──────────────────────────────────────────┼──────────────────────────────┤
# │ N0   │ —                                        │ Control / baseline           │
# │ N1   │ CACR  (w=0.10)                           │ Does CACR alone help?        │
# │ N2   │ CE-AURC (w=0.01)                         │ Does CE-AURC alone help?     │
# │ N3   │ CACR + CE-AURC                           │ Do training-time mods stack? │
# │ N4   │ TENT  (steps=1, lr=0.001)                │ Does TENT alone help?        │
# │ N5   │ CACR + CE-AURC + TENT                    │ Does TENT add on top of N3?  │
# │ N6   │ CACR  w=0.05                             │ CACR sensitivity (gentler)   │
# │ N7   │ CACR  w=0.50                             │ CACR sensitivity (stronger)  │
# │ N8   │ CACR  w=0.10, neg_weight=0.10            │ Background-penalty variant   │
# │ N9   │ CE-AURC w=0.05                           │ CE-AURC sensitivity (high)   │
# │ N10  │ CE-AURC w=0.001                          │ CE-AURC sensitivity (low)    │
# │ N11  │ TENT steps=3                             │ TENT sensitivity (more)      │
# │ N12  │ TENT steps=5                             │ TENT sensitivity (aggr.)     │
# └──────┴──────────────────────────────────────────┴──────────────────────────────┘
#
# Total per architecture: 13 configs × 3 LOCO folds = 39 SLURM jobs.
# Architectures: MAMNet, OGLANet, DINOv3 → 117 jobs total upstream.
# This wrapper only schedules the post-hoc analysis (single CPU job).
#
# =====================================================================
# Server paths — uncomment the block for your target server
# =====================================================================

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

LOG_DIR="${BASE_PATH}/data/newmod_analysis_logs"
mkdir -p "${LOG_DIR}"

name="newmod_ablation_analysis"
outputfile="${LOG_DIR}/${name}.out"

sbatch --output="${outputfile}" \
       --job-name="${name}" \
       newmod_ablation_analysis.sh "$@"

echo ""
echo "================================================================"
echo "  Diagnostic-Module Ablation Analysis queued"
echo "================================================================"
echo ""
echo "  Configs analyzed (per architecture):"
echo "    N0   C4 baseline (control)"
echo "    N1   C4 + CACR (w=0.1)"
echo "    N2   C4 + CE-AURC (w=0.01)"
echo "    N3   C4 + CACR + CE-AURC                — training-time stack"
echo "    N4   C4 + TENT (steps=1)                — test-time adapter"
echo "    N5   C4 + CACR + CE-AURC + TENT         — full stack"
echo "    N6   CACR w=0.05    (sensitivity)"
echo "    N7   CACR w=0.50    (sensitivity)"
echo "    N8   CACR w=0.10, neg_weight=0.1"
echo "    N9   CE-AURC w=0.05 (sensitivity)"
echo "    N10  CE-AURC w=0.001 (sensitivity)"
echo "    N11  TENT steps=3   (sensitivity)"
echo "    N12  TENT steps=5   (sensitivity)"
echo ""
echo "  Decision tree the analysis answers:"
echo "    1.  Single-module effect:  N1, N2, N4 vs N0"
echo "    2.  Composition (do they stack?):  N3 vs N1⊕N2,  N5 vs N3⊕N4"
echo "    3.  Hyperparameter sensitivity:    N6/N7/N8, N9/N10, N11/N12"
echo "    4.  Per-architecture verdict:      best config per arch, with"
echo "                                       worst-case-safety check"
echo ""
echo "  Predictions verified against §5 design rationale:"
echo "    P1: CACR helps mean mIoU on majority of architectures"
echo "    P2: CE-AURC alone does not collapse mIoU on any architecture"
echo "    P3: TENT helps DINOv3 most (clean encoder + correctable decoder)"
echo "    P4: TENT does not catastrophically fail OGLANet (encoder-locus)"
echo "    P5: N5 full stack ≥ best single-module on majority of archs"
echo ""
echo "  Outputs:"
echo "    ${BASE_PATH}/data/newmod_analysis/table_newmod_ablations.tex"
echo "    ${BASE_PATH}/data/newmod_analysis/newmod_report.json"
echo "    ${BASE_PATH}/data/newmod_analysis/newmod_supplementary.json"
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Log:      tail -f ${outputfile}"
echo ""