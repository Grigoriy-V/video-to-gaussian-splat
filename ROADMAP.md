# Project Roadmap

**Updated:** 2026-09-01

**Project status:** first end-to-end baseline complete; controlled preprocessing
experiments are next

**Current approved step:** documentation and experiment definition only; no new
Modal deployment, upload, GPU run, or training is authorized by this roadmap

This is the canonical roadmap for `video-to-gaussian-splat`. It records current
state and experiment order. Listing work here does not authorize implementation,
model downloads, external data transfer, deployment, GPU use, training, commit,
or publication.

## Project direction

The repository currently reconstructs a static 3D Gaussian Splat from an
ordered RGB sequence:

```text
video or generated orbit
  -> frames
  -> COLMAP camera solve
  -> Splatfacto optimization
  -> Gaussian Splat PLY
```

The active research input is AI-generated camera coverage derived from one or
more image references. Ordinary captured and rendered videos remain valid
inputs, and the reconstruction runtime must not depend on a particular video
generator.

The longer-term goal is not only `video -> splat`. Reconstructed splats should
become reusable 3D priors inside a generative pipeline:

```text
image references
  -> generated views/video
  -> reconstructed Gaussian Splat
  -> camera-controlled RGB/depth/normal/alpha renders
  -> conditioning or structural reference for new image/video generation
  -> improved coverage and reconstruction
```

This feedback loop is a research direction, not a currently implemented
feature. A splat render may provide spatially consistent camera views and
geometry-derived controls, but whether a specific image/video model can consume
them effectively must be demonstrated rather than assumed.

## Completed foundation

- Independent repository and private GitHub remote created.
- Batch-only Modal App `gaussian-splat-trainer` deployed.
- Nvidia L4 runtime verified with Nerfstudio 1.1.5, gsplat 1.4.0, and COLMAP
  3.9.1.
- Scale-to-zero operation verified; there is no web endpoint or warm GPU pool.
- Input manifest, dimensions, run identity, non-overwrite behavior, camera
  registration gate, training, export, and CPU-only status paths implemented.
- COLMAP preparation made robust to multiple sparse components by selecting the
  model with the greatest number of registered images.

## Experiment 01: native unmasked baseline -- complete

Configuration:

```text
94 generated RGB frames
768 x 960
no foreground masks
Splatfacto default strategy
fixed white training background
30,000 steps
Nvidia L4
```

Result:

- COLMAP registered 94/94 frames;
- camera preparation took 89.8 seconds;
- training took 830.3 seconds;
- final reported train loss was 0.0118;
- export contained 85,834 Gaussians;
- the character was coherent and recognizable across novel views;
- residual floating Gaussians remained above the head and around the shoulders.

Decision: the core hypothesis passed. Generated orbit video can be consistent
enough for a complete camera solve and recognizable static 3DGS. Preserve this
run unchanged as the baseline.

Evidence: `reports/experiment-01-h3-orbit-3dgs.md`.

## Experiment 02: native resolution plus foreground mask -- run complete

Purpose: isolate the effect of foreground masking on floating artifacts and
subject boundaries.

Planned comparison:

```text
native RGB
+ accepted BiRefNet masks at the same resolution
+ accepted camera solution
+ Splatfacto default strategy
-> smoke
-> 30,000 steps only after smoke acceptance
```

The current local preprocessing candidate contains 50 RGBA frames at 768 x 960.
The alpha channel was produced with BiRefNet and VITMatte in local ComfyUI. The
mask must be exported as a one-channel black/white PNG for every RGB frame.
Black pixels are ignored; white pixels participate in training.

Before training:

1. record the RGB and alpha sequence manifests;
2. verify frame correspondence and dimensions;
3. inspect frontal, profile, rear, hair, glasses, shirt, shoulders, and crop
   boundaries;
4. generate mask statistics and a contact sheet;
5. freeze mask conversion settings and a new job/run identity;
6. decide whether to reuse an accepted camera solution or solve this 50-frame
   sequence separately.

Acceptance is comparative and visual: retain subject detail while materially
reducing background and floating Gaussians. Do not claim improvement from train
loss alone.

The frozen BiRefNet binary-mask run completed on 2026-09-01: 94/94 masks were
accepted, the 30,000-step L4 optimization took 715.78 seconds, and the export
contains 69,578 Gaussians in a 16.46 MiB PLY. Visual A/B rejected the result:
overall quality and edge stability were worse, while the overhead floaters and
white silhouette fringe remained. The unmasked white-background run remains
the accepted baseline. Full evidence and the run-log limitation are recorded
in `reports/experiment-02-birefnet-mask-3dgs.md`.

## Experiment 03: x2 upscale plus the same mask -- after Experiment 02

Purpose: isolate whether source upscaling improves the splat or merely introduces
sharper but inconsistent details.

Current local candidate:

```text
upscaler: 2xLiveActionV1_SPAN_490000.pth
RGB output: 1536 x 1920
frames: 50
```

Controlled comparison:

```text
x2 RGB
+ the accepted native alpha resized deterministically to x2
+ thresholded one-channel black/white x2 masks
+ the same camera poses with intrinsics scaled x2
+ Splatfacto default strategy
-> new smoke run
```

Do not rerun segmentation on the upscaled frames in this first A/B; otherwise
both the image and mask generator change. A separate best-quality experiment
may regenerate masks after upscaling under a new identity.

Evaluate cross-view consistency around hair, eyes, glasses, skin, shirt edges,
and jacket texture. The x2 sequence has four times as many pixels, so measure
runtime and memory with a smoke before authorizing 30,000 steps.

## Experiment 04: MCMC strategy -- after mask and upscale baselines

Purpose: compare Splatfacto MCMC with the default densification strategy on an
accepted dataset configuration.

Freeze the input resolution, masks, camera poses, frame split, background, and
step count before changing strategy. MCMC receives a new run identity and must
first pass a bounded smoke. Compare:

- visible reconstruction quality from the same camera paths;
- floating and oversized Gaussian artifacts;
- Gaussian population and PLY size;
- training time and peak memory;
- held-out metrics if a frozen evaluation split has been introduced by then.

Do not combine the first MCMC test with a new upscaler or new mask policy.

## Experiment 05: improve splat optimization -- research track

After the controlled mask, resolution, and MCMC comparisons, investigate
training improvements one material variable at a time. Candidate areas:

- held-out view protocol with PSNR, SSIM, and LPIPS;
- foreground-aware losses and background treatment;
- mask erosion/dilation and uncertain-boundary handling;
- opacity initialization and pruning thresholds;
- densification schedule and Gaussian population limits;
- scale and anisotropy regularization for oversized floaters;
- camera-pose or focal-length refinement after COLMAP;
- frame selection based on sharpness, redundancy, and geometric consistency;
- robust handling or rejection of temporally inconsistent generated frames;
- initialization from alternative sparse points or geometry priors;
- conservative automatic cleanup with an unchanged raw export retained.

Each comparison needs an immutable config, smoke result, cost/timing record,
visual evidence, and a separate report. Avoid broad sweeps until individual
variables have shown useful signal.

## Preprocessing service -- separate future component

The proposed batch App `video-to-gaussian-splat-preprocess` remains independent
from both H3 generation and `gaussian-splat-trainer`. It uses its own data and
model Volumes and transfers only accepted artifacts through an explicit import.

The current design covers BiRefNet, SAM 3, SPAN/Real-ESRGAN/SwinIR, optional RTX
Video Super Resolution, validation, and contact sheets. Direct Python is the
preferred first backend; ComfyUI is optional for integrations that materially
benefit from it.

Design: `docs/preprocessing-app-design.md`.

## Generative feedback loop -- later research

Once a stable splat is available, investigate its use as a 3D-consistent source
for subsequent generation:

1. render known and novel camera trajectories from the splat;
2. export aligned RGB, alpha, depth, normals, and camera metadata where
   available;
3. test these renders as references or structural controls for image/video
   generation;
4. target missing or weakly reconstructed viewpoints;
5. validate generated additions before merging them into a new reconstruction;
6. compare the refined splat against the previous immutable baseline.

Potential outcomes include camera-controlled video generation, consistent image
sets, view completion, and iterative reconstruction. Keep generated evidence
separate from captured evidence and never treat hallucinated surfaces as ground
truth.

## Deferred work

- direct MP4 ingestion and frame extraction;
- alternative camera solvers;
- 4DGS or explicitly dynamic scenes;
- production cleanup/export tooling;
- public artifact or checkpoint release;
- integration with a specific hosted generation provider.

## How to maintain this file

After a material decision, update only the affected state. When an experiment
closes, record its concise result, evidence, decision, and limitation here; keep
commands, hashes, detailed metrics, and long analysis in `reports/`. Clear or
replace `Current approved step` explicitly. Never use roadmap presence as
execution authority.
