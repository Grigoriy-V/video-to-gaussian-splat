# Generated-input consistency research

**Status:** research notes and experiment candidates only. Nothing in this
document authorizes a model download, preprocessing deployment, GPU run, or
training change.

## Problem

AI-generated orbit video may look smooth while violating the static multi-view
assumption used by COLMAP and 3DGS. Facial features, glasses, hair, clothing,
silhouettes, lighting, or local geometry can drift between frames. Static 3DGS
then attempts to fit contradictory observations and may create floaters,
translucent duplicates, unstable boundaries, elongated Gaussians, or
view-dependent overfitting.

The research target has two levels:

```text
frame confidence: should this view participate in reconstruction?
pixel confidence: which parts of an accepted view are cross-view consistent?
```

These estimates are distinct from a semantic foreground mask. A foreground
mask answers where the subject is; an inconsistency map answers which
observations of that subject are supported by other views.

## Optical-flow candidates

### NVIDIA Optical Flow Accelerator

NVIDIA Optical Flow Accelerator (NVOFA) is a hardware optical-flow engine on
supported Turing, Ampere, Ada, and later GPUs. It runs independently of CUDA
cores and can provide forward/backward motion vectors at high throughput. The
local RTX 4090 and Modal L4 are Ada GPUs, but the preprocessing runtime must
still be tested for driver, SDK, library, and API availability.

Potential use:

- rapid flow for every adjacent frame pair;
- forward/backward cycle checks;
- NVOFA cost output as one reliability signal where exposed;
- frame-level anomaly scoring;
- low-cost batch preprocessing if boundary quality is sufficient.

Evaluate thin hair and glasses, occlusion boundaries, newly revealed surfaces,
and whether hardware cost is calibrated well enough for soft confidence.

Primary references:

- <https://developer.nvidia.com/optical-flow-sdk>
- <https://docs.nvidia.com/video-technologies/optical-flow-sdk/nvofa-programming-guide/index.html>

### Neural and classical alternatives

| Method | Intended role |
|---|---|
| NVOFA | high-throughput hardware baseline |
| Torchvision RAFT-Large | simple quality-oriented neural baseline |
| SEA-RAFT | efficient neural flow with uncertainty-oriented output |
| OpenCV DIS | lightweight software baseline |
| OpenCV Farneback | diagnostic lower bound only |
| UniMatch or GMFlow | later comparison if RAFT-family results are insufficient |

Primary references:

- <https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html>
- <https://github.com/princeton-vl/SEA-RAFT>

The first comparison should use a frozen set of difficult adjacent pairs that
covers frontal, profile, rear, hair, glasses, shoulder boundaries, and
front/back clip transitions. Compare flow visualization, cycle error, invalid
or occluded regions, runtime, memory, and the resulting confidence maps.

## Flow-based inconsistency

For adjacent frames `I0` and `I1`, estimate both directions:

```text
F01 = flow(I0, I1)
F10 = flow(I1, I0)
```

Warp `I1` into coordinates of `I0` using `F01`. Reject correspondences that
leave the image or fail the forward/backward cycle:

```text
cycle_error(x) = ||F01(x) + F10(x + F01(x))||
```

Only valid, non-occluded correspondences should contribute to inconsistency.
RGB residual alone is insufficient because lighting can drift; combine it with
feature-space residual where useful.

Prefer a soft confidence map over a binary mask:

```text
confidence(x) = exp(-inconsistency(x) / sigma)
```

For an internal frame, combine confidence from its previous and next
neighbours. Evaluate a conservative minimum and a geometric mean. Also test the
final-to-first pair when the orbit is expected to close.

Flow should first be estimated on the native `768 x 960` sequence before AI
upscaling. Resize confidence to `1536 x 1920` when needed. If flow vectors are
resized by two, multiply both horizontal and vertical magnitudes by two.

Optical flow does not prove rigid 3D consistency. A smoothly morphing face may
still yield plausible flow. Later tests should add longer-range tracks,
multi-frame cycle checks, features, COLMAP evidence, and depth consistency.

## Depth candidates

### Depth Anything V2

Depth Anything V2 is a monocular relative-depth baseline, but independent
frames may have different scale and shift. Do not subtract raw maps directly.
First align adjacent maps on reliable correspondences:

```text
D0 ~= a * warp(D1) + b
```

Compute depth residual only after robust scale-and-shift alignment and only
where optical-flow correspondence is valid.

Primary reference: <https://github.com/DepthAnything/Depth-Anything-V2>

### Video Depth Anything

Video Depth Anything is a candidate when temporal stability matters more than
independent monocular inference. It may reduce depth flicker, but its geometry
still requires validation against cameras and multi-view evidence.

### Depth Anything 3

Depth Anything 3 is the most relevant multi-view candidate because it can
consume multiple images and optionally known camera intrinsics and extrinsics.
A pose-conditioned test can provide accepted COLMAP cameras and request aligned
multi-view geometry instead of independently inferred depth.

Primary references:

- <https://github.com/ByteDance-Seed/Depth-Anything-3>
- <https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md>

Check model license, checkpoint size, runtime, camera conventions, and output
scale before integration or deployment.

## Ways depth may be used

Ordered from lowest to highest risk:

1. **Diagnostics only.** Produce depth, consistency maps, frame scores, and
   contact sheets without changing reconstruction.
2. **Frame filtering.** Reject unusually inconsistent views while preserving
   angular coverage.
3. **Soft pixel confidence.** Combine flow, RGB/feature, and aligned-depth
   confidence to down-weight locally contradictory supervision.
4. **Depth supervision.** Compare rendered splat depth with validated
   pseudo-depth only in high-confidence regions.
5. **Point initialization.** Back-project multi-view-confirmed depth into a
   filtered point cloud for Gaussian initialization.
6. **PLY analysis and cleanup.** Flag Gaussians whose depth is unsupported by
   several accepted views while retaining the immutable raw export.

Predicted depth is a prior, not ground truth. It may reproduce or introduce
AI-generated errors, especially on hair, glasses, white background, and weakly
observed rear surfaces.

## Proposed research sequence

Do not combine all signals in the first run:

```text
Phase 1: frozen difficult frame pairs
  -> NVOFA vs RAFT-Large or SEA-RAFT
  -> flow and cycle-error report

Phase 2: accepted cameras + native frames
  -> Video Depth Anything or pose-conditioned Depth Anything 3
  -> aligned depth-consistency report

Phase 3: flow + depth + COLMAP evidence
  -> per-frame scores and filtered sequence
  -> unchanged Splatfacto A/B

Phase 4: accepted full sequence
  -> soft per-pixel confidence
  -> robust or weighted loss A/B

Phase 5: validated depth prior
  -> depth loss or point initialization A/B
```

For each phase, preserve the input manifest and camera coverage, record model
and checkpoint identity, and generate visual evidence. Do not select thresholds
from final splat views later used to claim improvement.

