# Experiment 03: x2 upscaled generated input

**Status:** 30,000-step result accepted as a practical quality improvement;
the attempted 40,000-step resume is invalid because model tensors did not update.

**Date:** 2026-09-01

## Objective

Test whether local x2 upscaling of the generated orbit sequence improves the
practical visual quality of the reconstructed character. The project values the
visible production result over an exhaustive scientific separation of every
camera and resolution variable.

## Frozen input and configuration

```text
input: AI-generated character orbit, locally upscaled x2
upscaler: 2xLiveActionV1_SPAN_490000.pth
frames: 94 RGB PNG
resolution: 1536 x 1920
manifest: 1d00c10aaa40eb511398f469a6d269e34b3382d0d673506277544ff712af1564
masks: none
camera solve: new COLMAP solve on the x2 sequence
registered cameras: 94/94
method: Nerfstudio Splatfacto, default strategy
background: fixed white
GPU: Nvidia L4
requested training: one run to 40,000 steps
available valid checkpoint: 30,000 steps
accepted practical result: 30,000 steps
```

COLMAP preparation took 124.92 seconds and produced one accepted sparse model
with all 94 frames registered.

## Run interruption and recovery

Modal preempted the L4 container after the run had progressed beyond 30,000
steps. Checkpoints at 10k, 20k, and 30k survived. Modal then automatically
retried the same function input, but the repository's non-overwrite guard
stopped the retry because the run directory already existed.

The 30k checkpoint was exported independently. A later resume appeared to
produce `step-000040000.ckpt`, after which the container continued logging
beyond the requested step instead of finalizing. The invocation was cancelled
after that file was committed. A separate bounded export loaded it successfully,
but binary inspection subsequently proved that it was not a valid trained 40k
model.

The resume event contains steps beyond 40k (through 48,096), so those later
entries are treated as an orchestration/finalization anomaly and are excluded
from the experiment comparison. No GPU container remained active after the
cancel and export.

## 30k artifacts

| Artifact | Value |
|---|---|
| Checkpoint | `step-000030000.ckpt` |
| Checkpoint size | 75.9 MiB |
| Local PLY | `results/experiment-03-upscaled-x2-30k/splat-30k.ply` |
| Exported Gaussians | 97,502 |
| PLY size | 24,182,088 bytes / 23.06 MiB |
| PLY SHA-256 | `a42596bb5bfe8d964be58283b55899d51a708dfdae8475f6ba751ba20f55eb39` |
| Export time | 36.01 seconds |

## Invalid 40k resume: root-cause evidence

The exporter explicitly reported loading `step-000040000.ckpt`, but its PLY is
byte-for-byte identical to the accepted 30k export. Inspection of the PyTorch
checkpoint ZIP payloads found that all 40 tensor-storage entries are also
byte-for-byte identical between `step-000030000.ckpt` and
`step-000040000.ckpt`; only `data.pkl` metadata and the serialization ID differ.
The nominal 40k file therefore advanced checkpoint metadata without updating
the learned model or optimizer tensors.

| Artifact | Value |
|---|---|
| Invalid checkpoint | `step-000040000.ckpt` |
| Checkpoint size | 75.9 MiB |
| Local PLY | `results/experiment-03-upscaled-x2-40k/splat-40k.ply` |
| Exported Gaussians | 97,502 |
| PLY size | 24,182,088 bytes / 23.06 MiB |
| PLY SHA-256 | `a42596bb5bfe8d964be58283b55899d51a708dfdae8475f6ba751ba20f55eb39` |
| Identity versus 30k PLY | exact byte match |
| Checkpoint tensor storages versus 30k | 40/40 exact byte matches |

Metrics below are retained only as diagnostics of the failed resume. They must
not be reported as results from a valid 40k model. The unchanged all-view
metrics are consistent with the unchanged tensor payload.

| Metric | 30k | 40k | Change |
|---|---:|---:|---:|
| Train loss at sampled step | 0.013463 | 0.013237 | -0.000226 |
| Mean train loss in preceding 1k window | 0.012832 | 0.012562 | -0.000270 |
| Sampled train PSNR | 28.1855 | 28.7244 | +0.5389 dB |
| All-training-views PSNR | 31.8549 | 31.8549 | 0.0000 dB |
| All-training-views SSIM | 0.970212 | 0.970212 | 0.000000 |
| All-training-views LPIPS | 0.097252 | 0.097251 | approximately 0 |
| TensorBoard Gaussian count | 98,430 | 98,430 | 0 |
| GPU memory | 11,412 MiB | 11,653 MiB | +241 MiB |

The resume event took 1,023.9 seconds (17.1 minutes) from its first logged step
after 30k to nominal step 40k. The small sampled-loss changes reflect different
sampled training images and forward passes, not learned-parameter updates.

Two defects caused the misleading outcome:

1. The added resume path trusted Nerfstudio 1.1.5 Splatfacto checkpoint resume
   without verifying that model tensors changed. In this run the loop and loss
   logging advanced while the checkpoint tensor storages remained frozen.
2. Nerfstudio interprets `max_num_iterations` as additional iterations after
   `_start_step`, not as an absolute target step. Passing 40,000 while resuming
   at 30,000 therefore permitted progress toward 70,000; this explains the
   observed logging beyond 40k.

The repository now refuses in-place Splatfacto resume rather than silently
producing another false continuation. A valid 40k comparison requires a fresh
0-to-40k run under a new run identity unless a separately verified continuation
implementation is introduced.

## 30k training diagnostics

These values measure fit to training views because the run used
`eval_mode=all`. They are not held-out novel-view metrics.

| Metric | Value |
|---|---:|
| Train loss at step 30k | 0.013463 |
| Mean train loss, 29k-30k | 0.012832 |
| All-training-views PSNR | 31.8549 dB |
| All-training-views SSIM | 0.970212 |
| All-training-views LPIPS | 0.097252 |
| TensorBoard Gaussian count | 98,430 |
| GPU memory | 11,412 MiB |
| Event wall time to 30k | 2,952.8 seconds / 49.2 minutes |

All-training-view metrics continued to improve from 25k to 30k:

| Step | PSNR | SSIM | LPIPS |
|---:|---:|---:|---:|
| 25k | 31.4625 | 0.968900 | 0.099896 |
| 30k | 31.8549 | 0.970212 | 0.097252 |

The run had not reached a strict numerical plateau at 30k, but the practical
decision is based primarily on the inspected visual result.

## Visual result

The x2 splat is visually better than the native-resolution baseline in the
areas that matter most for the current application:

- higher perceived character detail;
- sharper facial features, glasses, hair, and clothing;
- better overall presentation and subject quality.

The x2 result also contains more floating outliers. This geometric cleanliness
regression is visible but is not currently severe enough to outweigh the
improvement in character quality.

LPIPS is worse than in the native baseline while SSIM is better and PSNR is
similar. This does not override the visual acceptance: the x2 targets contain
upscaler-created high-frequency details, and the native and x2 runs are not
fitted against identical target images. Floating artifacts may also be weakly
visible in training cameras while becoming clearer from free novel views.

## Practical decision

Accept x2 preprocessing as the preferred current input treatment. Do not spend
the next iteration on a strict native-versus-x2 camera-pose comparison or on a
separate run that reuses scaled native cameras. The expected practical
difference is not important enough for the current project goal.

Discard the nominal 40k continuation as invalid. Keep the 30k artifact as the
accepted checkpoint. Do not make a quality claim about 30k versus 40k from this
attempt; that question remains unanswered until a valid fresh 40k run exists.

The reconstruction is close to a successful production result. The remaining
material blocker is:

> Remove or substantially reduce the white fringe and residual white-background
> splats around the subject without sacrificing the accepted x2 detail.

Future work should therefore prioritize background-edge treatment. Preserve
the accepted x2 RGB and subject detail while investigating alpha decontamination,
foreground color de-spill, uncertain-boundary handling, background-aware loss,
or conservative post-training cleanup. The previously tested hard BiRefNet
mask is not accepted as the solution because it reduced overall quality and did
not remove the white fringe reliably.

## Deferred by decision

- strict scientific separation of upscaling from the new COLMAP solve;
- x2 training with reused native-resolution camera extrinsics;
- exhaustive native-versus-x2 metric normalization;
- broad camera and preprocessing sweeps before the white-fringe issue is solved.

The Blender-camera-prior experiment remains useful for future generated-video
control, but it is a separate pipeline experiment rather than a requirement for
accepting this x2 reconstruction.
