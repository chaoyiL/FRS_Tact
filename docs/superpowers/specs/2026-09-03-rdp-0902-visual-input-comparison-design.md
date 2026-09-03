# RDP 0902 Visual Input Comparison Design

## Goal

Create a compact, reproducible visual report comparing the inputs seen by the
0902 RDP checkpoint in its confirmed `insert_01` training source and in the
141807/142024 robot runs. The report must show both raw appearance differences
and differences after the 0902 visual encoder, without scanning every robot
frame or changing deployment behavior.

## Sample Contract

- Training: 15 frames from checkpoint-listed training episodes 0, 200, and 400.
- Robot 141807: the existing 16 representative frame IDs.
- Robot 142024: the existing 11 representative frame IDs.
- Decode all JPEG inputs as RGB and keep the production mapping:
  `camera0` is the physical left-hand view and `camera1` is the physical
  right-hand view; tactile suffix `_0` belongs to the left hand and `_1` to the
  right hand.
- Use the 0902 LDP, deployable AT, PCA, and encoder0824 artifacts with seed 0.

## Outputs

Write only under `outputs/rdp_0902_visual_comparison/`:

1. `input_montage.png`: representative training and robot images for all six
   visual/tactile streams, with source and frame labels.
2. `distribution_report.png`: RGB/luminance histograms, a two-dimensional PCA
   projection of 0902 visual features, and counterfactual action-magnitude bars
   for original, right-tactile replacement, right-visual replacement, and
   combined replacement.
3. `metrics.json`: sample identities and the numeric values used by the plots,
   including brightness, contrast, RGB means, edge magnitude, histogram
   Jensen-Shannon distance, feature centroid separation, and action magnitudes.

## Interpretation Boundaries

- Report right-visual and right-tactile replacement as sensitivity probes.
- A larger displacement is not called correct without a semantically matched
  ground-truth action.
- Note that 142024 contains an out-of-range wrist-state component.
- Do not recommend feeding static training images to a live robot.

## Verification

- Confirm both PNG files decode and have nonzero dimensions.
- Confirm `metrics.json` parses, contains finite numeric values, and records all
  42 sampled inputs.
- Confirm no robot bridge or network connection is constructed.
