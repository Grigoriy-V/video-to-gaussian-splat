# Experiment 02: BiRefNet foreground mask

**Run date:** 2026-09-01  
**Status:** complete; visually rejected

## Purpose

Measure the effect of foreground masking while keeping the accepted RGB frames,
camera solution, Splatfacto strategy, white background, and 30,000-step budget
unchanged from Experiment 01.

## Frozen input and configuration

| Item | Value |
|---|---|
| RGB frames | 94 PNG, 768 x 960 |
| RGB manifest | `0c50d4e1c8fc7735ebfae4fa71f78a80973d3d7fbed1efcb2cce73a3fff4ad2c` |
| Camera solve | reused accepted Experiment 01 solve, 94/94 registered |
| Masks | 94 one-channel binary PNG, 768 x 960 |
| Mask source | BiRefNet/VITMatte local ComfyUI output, threshold 0.5 |
| Mask manifest | `c2fcbcc6b6d888c597b18a89f08e0eeecc83a370e4317b05a9c29134dcc75225` |
| Mask identity | `birefnet-binary-v1` |
| Method | Nerfstudio 1.1.5 Splatfacto, default strategy |
| Background | fixed white |
| Split | `eval_mode=all`; no held-out views |
| GPU | Nvidia L4 |
| Main run ID | `splatfacto-white-mask-birefnet-main-v1` |

## Execution results

| Stage | Result |
|---|---|
| Mask attachment | 94/94 accepted; manifest verified |
| 2,000-step smoke | completed in 85.12 s |
| Smoke export | completed in 34.60 s |
| 30,000-step main | completed in 715.78 s |
| Main export | completed in 32.63 s |
| Final checkpoint | `step-000029999.ckpt` |
| Exported Gaussians | 69,578 |
| PLY size | 17,256,936 bytes (16.46 MiB) |
| PLY SHA-256 | `4a5f420f361f2d5058049c86c329d4a41e04edfe0ca9b3bd1784c1dfa595ba3b` |

Experiment 01 exported 85,834 Gaussians. The masked run exported 69,578,
approximately 18.9% fewer. This is a population/size observation, not yet proof
of better visual quality.

## TensorBoard metrics

The completed events were parsed after the run. Metrics below use the last
available sample for the same tag and step in each run.

| Metric | Baseline | Masked | Interpretation |
|---|---:|---:|---|
| Train loss, step 29,990 | 0.01146 | 0.00711 | lower masked optimization loss; domains differ |
| Train PSNR, step 29,990 | 29.56 dB | 22.21 dB | standard full-image training metric |
| All-images PSNR, step 29,000 | 31.91 dB | 24.12 dB | training views, not held-out views |
| All-images SSIM, step 29,000 | 0.9666 | 0.9387 | training views, not held-out views |
| All-images LPIPS, step 29,000 | 0.0832 | 0.1245 | training views, not held-out views |
| Gaussian count, step 29,990 | 86,661 | 69,969 | optimizer population before export |
| GPU memory, step 29,990 | 948.74 MB | 1,054.46 MB | TensorBoard-reported allocation |
| Nerfstudio train total | 804.04 s | 693.79 s | internal train timer |

The exported PLY counts differ slightly from the last logged optimizer counts:
85,834 baseline and 69,578 masked. The masked export is 18.9% smaller by count.

The lower masked loss is not directly comparable with the baseline loss because
background pixels no longer contribute to the optimized pixel loss. Conversely,
the standard PSNR/SSIM/LPIPS values are full-image metrics and penalize the
unreconstructed background. They therefore do not isolate foreground quality.
The next quantitative protocol should compute foreground-only and boundary-band
metrics with a frozen held-out split.

## Visual evaluation

The masked result was inspected interactively against the unmasked baseline and
was rejected:

- overall reconstruction quality is worse;
- subject edges drift or deform from some viewpoints;
- the floating artifacts above the head remain, although they are not the most
  important defect;
- the white fringe around the silhouette remains in some viewpoints;
- the smaller Gaussian population did not translate into a cleaner or more
  stable reconstruction.

Decision: do not promote the binary-mask configuration over Experiment 01. The
unmasked white-background run remains the accepted baseline.

The result suggests that a hard foreground mask alone does not remove white
edge contamination already present in RGB pixels and may amplify inconsistent
silhouettes between generated views. It also does not necessarily remove
floaters originating from inconsistent image content, sparse initialization,
or optimization behavior. These are hypotheses for later controlled tests, not
confirmed causal attributions.

## Artifact

Local PLY:

`results/experiment-02-mask-birefnet-main-v1/splatfacto-white-mask-birefnet-main-v1/splat.ply`

Modal PLY:

`/jobs/mix-back-clean-v1-largest-model/export/splatfacto-white-mask-birefnet-main-v1/splat.ply`

## Incident and limitation

The first main client call used a 120-second local timeout. The remote Modal
call continued correctly, but the missing report was initially misinterpreted
as cancellation and a duplicate main call was queued. `max_containers=1`
prevented concurrent GPU execution. The duplicate was cancelled before a
persistent run directory was created, while the original run continued to
30,000 steps.

Both calls used the old shared `logs/train-main.log` path. The duplicate opened
that path for writing, so the completed run's detailed text log was lost. Its
metrics were recovered from the intact TensorBoard event. The final checkpoint,
run config, event, report, and exported PLY are intact. Runtime code was
subsequently changed to use run-specific log and report filenames.

No PSNR, SSIM, or LPIPS is reported because the dataset used `eval_mode=all`.
Future mask work should not repeat this configuration unchanged. Candidate
controlled variants are soft-alpha compositing before training, conservative
mask erosion to exclude contaminated white boundary pixels, and a loss that
weights uncertain boundary pixels rather than treating them as hard binary
foreground. Each requires a separate run identity.
