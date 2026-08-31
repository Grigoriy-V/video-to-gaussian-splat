# Experiment 01: H3 orbit video to static 3D Gaussian Splat

## Outcome

The first end-to-end experiment succeeded. A synthetic 360-degree character
orbit generated with MiniMax H3 was treated as a multi-view capture of one
static subject. COLMAP registered all input frames, Nerfstudio Splatfacto
completed a 30,000-step optimization on a Modal Nvidia L4, and the exported PLY
produces a coherent novel-view character reconstruction.

The result is suitable as a proof of concept, but not yet as a clean production
asset. The head, face, glasses, hair, jacket, and profile remain recognizable
across views. Residual floating Gaussians remain above the head and around the
shoulders. These artifacts are consistent with unmasked white-background pixels
and small cross-frame inconsistencies in the generated video.

## Experiment identity

| Field | Value |
|---|---|
| Input | MiniMax H3 generated character orbit |
| Source directory | `D:/ML/Ai_render/projects/spot/refs/Mix_back_clean` |
| Frames | 94 PNG images |
| Resolution | 768 x 960 |
| Total input size | 36,301,564 bytes (34.62 MiB) |
| Sequence manifest SHA-256 | `0c50d4e1c8fc7735ebfae4fa71f78a80973d3d7fbed1efcb2cce73a3fff4ad2c` |
| Modal app | `gaussian-splat-trainer` |
| Modal environment | `grigoriy-v/main` |
| Modal volume | `gaussian-splat-runs` |
| GPU | Nvidia L4 |
| Job ID | `mix-back-clean-v1-largest-model` |
| Training run ID | `splatfacto-white-main-v1` |
| Configuration | `configs/mix_back_clean_v1_largest_model_white.json` |

No masks were used. MCMC was not used. This is the ordinary Splatfacto default
strategy baseline with a fixed white training background.

## Preparation and camera solve

| Metric | Result |
|---|---:|
| SfM implementation | COLMAP 3.9.1 |
| Camera model | `simple_pinhole` |
| Matcher | exhaustive |
| Registered frames | 94 / 94 (100%) |
| Acceptance threshold | 80 / 94 |
| Camera-solve status | accepted |
| Preparation elapsed time | 89.8 s (1 min 29.8 s) |

The initial preparation attempt exposed an integration issue rather than an
SfM failure: COLMAP produced two sparse components, while the conversion step
selected component `sparse/0`, which contained only 4 frames. Inspection of the
COLMAP database showed all 94 images, a median of 1,101 keypoints per image,
1,048 verified image pairs, and a connected match graph. The alternate sparse
component contained all 94 frames.

The preparation code was therefore changed to inspect every COLMAP sparse
model and convert the model with the greatest registered-image count. The
accepted run registered all 94 images. Sparse component numbering is not a
stable quality ranking, so largest-model selection remains part of the
reproducible preparation pipeline.

## Training configuration

| Field | Value |
|---|---|
| Framework | Nerfstudio 1.1.5 |
| Method | `splatfacto` |
| gsplat | 1.4.0 |
| Strategy | `default` |
| Iterations | 30,000 |
| Training/evaluation split | `eval_mode=all` |
| Downscale factor | 1 |
| COLMAP sparse points initialization | enabled |
| Background color | fixed white |
| Masks | none |
| Training elapsed time | 830.3 s (13 min 50.3 s) |
| End-of-run iteration time | 19.815 ms at step 29,990 |
| End-of-run throughput | 37.30 M rays/s at step 29,990 |
| Final reported train loss | 0.0118 |
| Last observed post-densification population | 86,661 Gaussians at step 14,900 |
| Final checkpoint | `step-000029999.ckpt` |
| Checkpoint size | approximately 68.0 MiB |

The earlier random-background smoke run reconstructed the subject but created a
large cloud of white Gaussians. The main run fixed the renderer background to
white. This substantially reduced the background cloud without introducing a
mask, while preserving the character reconstruction.

## Exported artifact

| Metric | Result |
|---|---:|
| Export format | binary little-endian PLY |
| Exported Gaussians | 85,834 |
| PLY size | 21,288,424 bytes (20.30 MiB) |
| Export elapsed time | 36.1 s |
| PLY SHA-256 | `7e9885ff79ff5a7adf6d7bab6752ddf4de6e7a303e06e17da1200e4002433c87` |
| Local artifact | `results/mix-back-clean-v1-largest-model/main-white/splat.ply` |
| Modal artifact | `/runs/jobs/mix-back-clean-v1-largest-model/export/splatfacto-white-main-v1/splat.ply` |

The 86,661 count is an intermediate observation at the last densification
event, not a final checkpoint count. It therefore should not be compared
directly with the 85,834 Gaussians serialized by the exporter.

## Timing and approximate cost

| Stage | Measured time |
|---|---:|
| Camera preparation | 89.8 s |
| Main training | 830.3 s |
| PLY export | 36.1 s |
| Measured stage total | 956.2 s (15 min 56.2 s) |

The previously observed Modal billing estimate was approximately USD 0.19 for
the L4 training job and about USD 0.01 for export. This is an estimate, not a
reconciled billing record, and excludes preparation, container startup, storage,
and data transfer. The deployment is configured to scale to zero with no warm
GPU container after a job finishes.

## Metrics that were not measured

PSNR, SSIM, and LPIPS are not available for this test. All 94 frames were used
for optimization (`eval_mode=all`), so there was no held-out evaluation set and
no defensible novel-view image metric. The final train loss describes fit to
the training sequence; it must not be interpreted as a generalization score.

The first test also did not measure camera reprojection error, foreground-mask
quality, geometry accuracy against ground truth, temporal identity drift, or a
quantitative artifact rate. Visual inspection in a splat viewer is therefore
the primary result evidence.

## Decision

The core hypothesis passed: a MiniMax H3 synthetic orbit can provide enough
multi-view consistency for COLMAP to solve a complete camera trajectory and
for ordinary static 3DGS to reconstruct a recognizable character.

The baseline should be retained unchanged. The next experiments should receive
new run identities and compare one variable at a time. The most useful next
A/B test is foreground masking, followed by conservative automatic PLY cleanup.
A held-out frame protocol should be introduced before comparing reconstruction
methods or claiming an improvement from PSNR, SSIM, or LPIPS.
