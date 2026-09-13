# Video to Gaussian Splat working contract

## Project boundary

- Keep this repository independent from `open-vocabulary-3d-vision-lab`,
  `Ai_render`, and any particular video-generation model.
- The product contract is ordered RGB frames -> COLMAP camera solve -> static
  RGB Gaussian Splat -> PLY.
- Input may originate from real video, rendered video, or generated video, but
  the current code accepts an already extracted PNG sequence rather than MP4.
- Assume a predominantly static scene. Do not describe this pipeline as 4DGS or
  as reconstruction of unobserved ground-truth geometry.
- Prefer small, reproducible batch functions over services or orchestration
  layers.

## Current research direction

The active research topic is AI image/reference -> generated orbit video ->
camera solve -> static 3D Gaussian Splat. The repository should support studying
the complete chain, not only the final Splatfacto invocation.

Relevant research and engineering work may include generative camera coverage,
multi-reference anchoring, temporal and geometric consistency, frame selection,
camera solvers, background treatment, masks, reconstruction strategies,
artifact cleanup, and defensible evaluation protocols. Keep comparisons
reproducible and change one material variable at a time.

Do not couple the core runtime to MiniMax H3 or any single generation provider.
Generated video is the current primary experimental input; ordinary captured
and rendered sequences remain valid inputs to the same reconstruction pipeline.

## Current implementation

- Modal App: `gaussian-splat-trainer`.
- Modal Volume: `gaussian-splat-runs`.
- Runtime image: `ghcr.io/nerfstudio-project/nerfstudio:1.1.5`.
- GPU: Nvidia L4.
- Method: Nerfstudio Splatfacto with gsplat.
- Camera preparation: COLMAP with automatic selection of the sparse model that
  has the greatest registered-image count.
- Local frame preparation includes `tools/select_motion_frames.py` for
  reproducible motion-weighted subsampling of an extracted image sequence.
- Control path: Modal SDK named functions; there is no HTTP or web endpoint.
- GPU functions use `min_containers=0`, `buffer_containers=0`,
  `max_containers=1`, `scaledown_window=2`, and `retries=0`.
- `status` is CPU-only. Do not add a warm GPU pool or `app.cls` model service
  without an explicit new requirement.
- `attach-rgba` can create an immutable RGBA dataset by reusing a compatible,
  accepted camera solution. RGBA and separate binary masks are mutually
  exclusive.
- Do not resume a Nerfstudio 1.1.5 Splatfacto run in place for step-count
  comparisons; a verified resume attempt left the Gaussian tensors unchanged.

## Experiment state

Experiments 01 through 04 are complete. The unmasked baseline used 94
768 x 960 H3-generated frames, registered 94/94 cameras, trained ordinary
Splatfacto for 30,000 steps with a fixed white background, and exported 85,834
Gaussians. It demonstrated a coherent reconstruction with residual background
artifacts. The hard binary-mask comparison was rejected visually. The x2
unmasked comparison improved visible detail but retained the white fringe.

The accepted foreground/background treatment is Experiment 04:
`configs/mix_back_clean_v3_soft_alpha_random.json`. It attached 94
native-resolution soft-alpha RGBA frames to the accepted camera solution,
trained for 30,000 steps with a random background, and exported 68,899
Gaussians. Visual inspection found that the white fringe and material
background outliers were removed. Full evidence is in
`reports/experiment-04-soft-alpha-rgba-3dgs.md`.

Do not silently reinterpret the reported train loss as a validation metric.
PSNR, SSIM, and LPIPS were not measured because the run used `eval_mode=all`
with no held-out frames.

The following remain separate future experiments, not baseline features:

- MCMC strategy;
- automatic PLY cleanup;
- direct video/frame extraction;
- held-out novel-view evaluation;
- alternative camera solvers or reconstruction methods.

## Execution and human gates

Work directly as one project agent. Do not delegate or create subagents. Own
the approved step from inspection through verification and reporting.

Discussion, documentation, or a plan does not authorize external or costly
actions. Treat each of the following as a separate explicit human gate:

- external downloads;
- input upload or other data transfer;
- Modal deployment;
- any GPU function call, including doctor, prepare, train, and export;
- main training or comparative runs;
- deletion of local or Modal data;
- Git initialization or history changes;
- remote creation, push, publication, or artifact release.

Read-only inspection, offline unit tests, syntax checks, and local report
generation are allowed within an approved implementation step.

## Run identity and artifacts

- Never overwrite a preparation, training run, export, or report.
- A changed input sequence or camera solve receives a new `job_id`.
- A changed method, strategy, background, mask policy, step count, split, or
  important training option receives a new run ID.
- Validate the expected image count, dimensions, and manifest before remote
  processing.
- Inspect the camera trajectory and sparse reconstruction after the automatic
  registration gate; a high registered-frame count alone is not proof of a
  correct solve.
- Keep source frames, checkpoints, PLY files, logs, credentials, caches, and
  Modal data out of Git unless the human explicitly approves a reviewed
  artifact for publication.

## Evaluation and claims

- Distinguish camera-solve metrics, training diagnostics, and held-out image
  metrics.
- Train loss measures fit to training images and is not a novel-view
  generalization metric.
- Report PSNR, SSIM, or LPIPS only when a frozen held-out split was actually
  evaluated.
- Record the data identity, configuration, registered-frame count, timing,
  checkpoint/export identity, artifact hash, visual limitations, and skipped
  metrics for each material experiment.
- Generated-video consistency is an experiment outcome, not an assumption.
- Preserve the completed baseline unchanged and compare one material variable
  at a time under new identities.

## Verification and reporting

Run checks in proportion to the change. Documentation-only changes require
link, formatting, and diff inspection rather than GPU or integration runs.
Code/config changes require the relevant offline unit tests and command
validation before any proposed external run.

After material work, update or add a report under `reports/`. Final handoff must
state changed files, checks, external actions and cost, limitations, and the
next human gate.
