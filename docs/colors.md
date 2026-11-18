# colors.md — the revision's fixed colour assignments

CONVENTIONS.md requires one colour per condition, fixed at first appearance and
reused so a reader comparing the revision against the submitted version does not
have to relearn the palette. This file did not exist when the revision started;
the assignments below were **measured from the submitted paper's own figure
panels** under `$PROJECT_ROOT/data/`, not
chosen fresh. Every new figure uses them.

## Per-architecture

**Revised 2026-07-31.** The submitted paper's architecture hues were kept as hue
families but re-laddered in luminance. The submitted values collided with the
city palette (OGLANet against Phoenix, MAMNet against Miami) and sat within
7.8 percentage points of relative luminance of each other, so a figure using
them was unreadable in greyscale and invited a reader to attach meaning to a
hue that carried two things at once. The hue angle of each architecture is
unchanged from the submitted panel, so a reviewer comparing versions still sees
teal-MAMNet, amber-OGLANet, violet-DINOv3.

| Architecture | Hex (current) | Submitted | Hue family | L\* |
|---|---|---|---|---|
| DINOv3 | `#664c94` violet | `#9881ba` | unchanged | 38 |
| MAMNet | `#259a9b` teal | `#69b1b1` | unchanged | 58 |
| OGLANet | `#eeb76b` amber | `#e0aa5f` | unchanged | 78 |

The submitted hues were measured from `final_figs/panel_b_bottom_ce_vs_01.png`
(submitted Fig 3(b), CE-vs-0/1 AURC scatter), the only submitted panel colouring
all three architectures at once, by sampling the legend swatch pixels.

Measured separation of the current triple, against the submitted one:

| Property | Submitted | Current |
|---|---|---|
| min pairwise gap in relative luminance (greyscale legibility) | 7.8 pp | **15.7 pp** |
| min pairwise L\* gap | 5.4 | **19.8** |
| min pairwise ΔE under simulated deuteranopia | 20.1 | **30.8** |
| min pairwise ΔE under simulated protanopia | 30.5 | **44.6** |

CVD simulation is the Viénot (1999) linear LMS approximation; ΔE is CIE76 in
CIELAB. Two alternatives were measured and rejected: CONVENTIONS.md's
tab:blue/orange/green order collapses under protanopia (min ΔE 5.8), and
Okabe-Ito separates well but discards the submitted paper's hue families, which
CONVENTIONS.md asks new figures to keep.

## Per-city

Measured from `figure3_panels/panel_c_intensity_ridge.png` (submitted Fig 4(E),
per-city grayscale-intensity ridge).

| City | Hex | Source panel |
|---|---|---|
| Chicago | `#4b3fd8` indigo | `panel_c_intensity_ridge.png` legend swatch |
| Phoenix | `#e8860a` orange | `panel_c_intensity_ridge.png` legend swatch |
| Miami | `#0abfa8` teal-green | `panel_c_intensity_ridge.png` legend swatch |

**One hue, one meaning, per figure.** In any figure that shows both, colour
carries architecture and cities are carried by position and axis labels only,
never by hue or fill. The city palette above stays for figures whose only
categorical axis is city. The collision that motivated the 2026-07-31 revision
was between the two systems: submitted-OGLANet `#e0aa5f` sat ΔE 31.7 from
Phoenix `#e8860a`, and submitted-MAMNet `#69b1b1` ΔE 23.8 from Miami `#0abfa8`.
The re-laddered architecture colours widen those to ΔE 33.3 and 21.3, which does
not on its own fix the ambiguity; the rule that colour means architecture and
nothing else is what fixes it. Figure 15 follows that rule.

## Roles

| Role | Value |
|---|---|
| baseline / comparator line | `#8c8c8c` grey, dashed or solid per CONVENTIONS.md |
| no-change or tolerance band | `#dedede` grey fill, no edge |
| zero / parity reference | `#b0b0b0` grey, dotted |
| marker edge, error bars | `#333333` |
| background | white |

## Marker conventions carried across revision figures

| Meaning | Encoding |
|---|---|
| condition has 2+ seeds, spread measurable | filled marker or solid bar, error bar drawn |
| condition has 1 seed, spread not measurable | open marker (white fill) or hatched bar, no error bar |
| excluded / collapsed run | never plotted, never in a summary; named in the caption |

## Other roles from the submitted figures, recorded so they are not reused

| Role | Hex | Source |
|---|---|---|
| foreground-class / SIB accent (coral-red) | `#e75e5e`, `#e55050` | `panel_b_class_asymmetry.png`, `panel_a_main_riskcoverage.png` |
| background-class (near-black) | `#2f3640` | `panel_b_class_asymmetry.png` |
| decoder retrain, Exp A | `#585ca4` | `panel_c_recovery_locus.png` |
| encoder retrain, Exp B | `#e9a847` | `panel_c_recovery_locus.png` |

The coral-red is the paper's alarm/foreground hue and is not available as a
data-series colour in a new figure.

## CrossEarth (added 2026-11-01, operator-approved)

CrossEarth joined the project after the three-architecture palette was fixed. It
uses **`#8c6d31` brown**, first used in the S-EO appendix figure draft.

| Architecture | Hex | L\* | Relative luminance |
|---|---|---|---|
| CrossEarth | `#8c6d31` brown | 47.9 | 16.7% |

Measured against the existing triple: ΔL\* 9.8 (DINOv3), 10.0 (MAMNet), 30.0
(OGLANet); CIE76 ΔE 77.1, 59.8, 31.8; relative-luminance gap 6.6, 9.1, 36.4 pp.
Hue separation is large, but the greyscale gap to DINOv3 (6.6 pp) is below the
15.7 pp floor of the triple. Rule: **every CrossEarth series uses a diamond
marker (`D`)**, and the three original architectures keep circles, so CrossEarth
stays distinguishable in greyscale and under colour-vision deficiency.
