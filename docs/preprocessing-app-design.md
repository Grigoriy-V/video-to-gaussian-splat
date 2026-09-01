# Preprocessing App: masks and upscaling

## Status

This document records the current design discussion and the first local ComfyUI
proof of concept. It is a proposal and experiment plan, not authorization to
deploy, download models, upload data, or run a cloud GPU.

The local proof of concept demonstrates that both foreground-mask generation
and image upscaling can be assembled before building a Modal service. Output
quality and sequence consistency have not yet been evaluated formally.

## Research objective

The reconstruction project is studying the complete pipeline:

```text
AI image or sparse references
  -> AI-generated orbit video
  -> frame preprocessing
  -> camera reconstruction
  -> static 3D Gaussian Splat
```

Preprocessing is a separate research stage. Its purpose is to test whether
foreground masks and higher-resolution source frames reduce floating Gaussians
and improve subject detail without introducing cross-view inconsistencies.

The principal risk is that an upscaler may produce sharper individual images
while inventing slightly different hair, glasses, skin, or clothing details in
each view. Such results can look better as video frames but be worse inputs for
static 3DGS.

## Local ComfyUI proof of concept

### Upscaling

The first local workflow used:

```text
model: 2xLiveActionV1_SPAN_490000.pth
operation: model-based image upscale
scale: x2, as encoded by the selected model
runtime: local ComfyUI
```

This establishes a concrete first upscaled sequence candidate. Record the exact
ComfyUI workflow JSON, node-pack versions, input sequence manifest, output
dimensions, output sequence manifest, runtime, and representative comparisons
before using it as an experiment input.

### Foreground masks

The first local workflow used the LayerStyle Advance BiRefNet nodes with the
following visible configuration:

```text
model: BiRefNet-general-epoch_244.pth
refinement method: VITMatte
detail erosion: 4
detail dilation: 4
black point: 0.01
white point: 0.99
process detail: false
device: CPU
maximum megapixels: 2.0
runtime: local ComfyUI
```

This is the initial mask candidate, not yet the frozen mask baseline. Before
training, inspect masks at the frontal, profile, rear, hair, glasses, white
shirt, shoulders, and frame-crop boundaries.

## Deployment boundary

Preprocessing should be a third independent Modal App:

```text
video-to-gaussian-splat-preprocess
```

It must remain separate from both existing applications:

- `minimax-h3-gpu`: generated-video production;
- `gaussian-splat-trainer`: COLMAP, Splatfacto training, and PLY export.

The preprocessing App must not reuse the H3 service class, H3 ComfyUI image, H3
model Volume, or the trainer data Volume. Reusing small implementation patterns
is acceptable; sharing runtime state is not.

### Independent storage

Proposed Volumes:

```text
video-to-gaussian-splat-preprocess-data
video-to-gaussian-splat-preprocess-models
```

Do not mount or write to `gaussian-splat-runs` from the preprocessing App.
Accepted preprocessing artifacts are transferred to the trainer only through a
separate explicit import action. This keeps source data, derived datasets, cost,
and failure recovery independent.

Suggested preprocessing layout:

```text
/jobs/<preprocess-job-id>/
  input/
  masks/
    birefnet-general-244-vitmatte-v1/
  upscaled/
    2xliveaction-span-x2-v1/
  combined/
    2xliveaction-span-x2-birefnet-v1/
  previews/
  reports/
```

Each derived directory receives a unique identity. Never overwrite masks or
upscaled frames when a model, checkpoint, threshold, refinement setting,
resolution, or source sequence changes.

## Proposed Modal architecture

The first deployment should remain batch-only and scale to zero:

```text
GPU: Nvidia L4
min_containers: 0
buffer_containers: 0
max_containers: 1
scaledown_window: 2
retries: 0
web endpoint: none
```

One App may expose several functions backed by separate pinned container images:

```text
preprocess App
  |- mask_birefnet
  |- mask_sam3
  |- upscale_span
  |- upscale_realesrgan
  |- upscale_swinir
  |- upscale_rtx_vsr
  |- validate
  |- contact_sheet
  `- status
```

This list is a research surface, not the required scope of the first deploy.
The smallest useful first version is BiRefNet plus the already tested SPAN x2
model, validation, and reports.

### Direct Python first

ComfyUI is not required for the first Modal implementation. BiRefNet and common
SPAN/ESRGAN/SwinIR inference paths can run directly under Python/PyTorch. A
direct backend provides a smaller image, simpler batch I/O, explicit model
identity, and easier validation.

SAM 3 should use a separate pinned image if its Python, PyTorch, or CUDA
requirements conflict with the BiRefNet/upscale environment.

### Optional ComfyUI image

A separate minimal ComfyUI image remains useful for models whose maintained
integration is primarily a ComfyUI node. It must not be based on or coupled to
the pinned H3 ComfyUI deployment.

The main candidate is NVIDIA RTX Video Super Resolution. NVIDIA provides an RTX
Video SDK and a ComfyUI integration, but Modal L4, Linux driver, SDK runtime,
and licensing compatibility require a bounded one-frame smoke before accepting
it as an available backend.

DLSS Super Resolution is not a candidate for generated RGB frames. It is a
renderer integration that expects render-time inputs such as motion vectors,
depth, exposure, and jitter history; the H3 sequence contains only RGB frames.

## Processing order

Keep the original RGB sequence immutable. Generate masks at native resolution
first:

```text
native RGB
  |- BiRefNet -> native masks
  `- upscaler -> x2 RGB

native masks -> deterministic x2 mask resize -> x2 masks
```

For a controlled upscaling A/B, do not run a second segmentation model on the
upscaled images. Reuse the accepted native masks and resize them with a recorded
method. Otherwise both segmentation and resolution change in the same test.

A later best-quality experiment may deliberately regenerate masks after
upscaling, but it must receive a separate identity.

## Validation

Every preprocessing run should produce a machine-readable report with:

- source and output manifest SHA-256;
- frame count and dimensions;
- model/checkpoint identity and source revision;
- complete inference parameters;
- runtime image identity;
- elapsed time and GPU type;
- output path and non-overwrite identity;
- failed or missing frames.

Mask validation should measure:

- foreground area per frame;
- frame-to-frame area change;
- number and size of connected components;
- contact with image borders;
- holes inside the principal component;
- empty and full-mask failures.

Visual evidence should include a contact sheet with, for selected frames:

```text
original | mask | red contour overlay | checkerboard composite
```

Upscale validation should include same-frame crops and adjacent-frame checks for
hair, glasses, eyes, skin, shirt edges, and jacket texture. Sharpness alone is
not an acceptance criterion; cross-view consistency is the important property
for 3DGS.

## Controlled reconstruction experiments

Retain Experiment 01 as the unmasked native-resolution baseline.

### Experiment 02: mask effect

```text
native 768 x 960 RGB
+ accepted BiRefNet native masks
+ existing accepted camera solution
-> Splatfacto smoke
-> Splatfacto 30k only after smoke acceptance
```

This isolates the effect of foreground masking.

### Experiment 03: first x2 upscale

```text
2xLiveActionV1 SPAN x2 RGB, 1536 x 1920
+ the same accepted masks resized x2
+ the same camera poses with intrinsics scaled x2
-> new Splatfacto smoke
```

This isolates the effect of the currently tested upscaler. The x2 images contain
four times as many pixels, so runtime and memory must be measured rather than
assumed from the native run.

### Later comparisons

Potential independent backends:

- Real-ESRGAN x2: simple direct-Python baseline;
- SwinIR x2: conservative image-restoration baseline;
- RTX Video Super Resolution x2: NVIDIA video-oriented candidate, conditional
  on compatibility smoke;
- SeedVR2 x2: heavier temporal video restoration candidate;
- SAM 3/3.1: temporally propagated foreground masks.

Do not enable face restoration such as GFPGAN or CodeFormer in the initial
upscale tests. Identity-changing or view-dependent facial enhancement is a
specific later experiment.

## Human gates

The following remain separate approvals:

1. implementation in the repository;
2. model/checkpoint download and license acceptance;
3. Modal Volume creation and data upload;
4. Modal deployment;
5. compatibility or smoke GPU run;
6. processing the complete sequence;
7. import into `gaussian-splat-trainer` storage;
8. masked or upscaled Splatfacto smoke;
9. 30,000-step training;
10. commit, push, or publication of code, workflows, reports, or artifacts.

## Immediate next decision

Before implementing the cloud App, preserve and evaluate the local ComfyUI
outputs:

1. export the exact workflow JSON;
2. record all installed node and model versions;
3. save the native mask sequence and x2 RGB sequence under immutable names;
4. generate contact sheets and mask statistics;
5. decide whether the first cloud backend must reproduce the local SPAN and
   LayerStyle workflow exactly or only reproduce its functional outputs through
   direct Python.
