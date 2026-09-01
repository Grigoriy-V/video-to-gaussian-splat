# Blender camera prior for generated-video reconstruction

**Status:** research design and candidate experiment only. This document does
not authorize Blender work, generation, data transfer, a COLMAP runtime
upgrade, deployment, or a GPU run.

## Research question

One proposed input is an AI-generated video conditioned by a Blender blockout
with a known camera trajectory. This creates two camera estimates:

```text
intended camera: Blender trajectory used by the blockout
observed camera: COLMAP trajectory reconstructed from generated frames
```

The experiment should measure whether the video generator preserves the
blockout camera and whether the Blender trajectory can improve reconstruction.

Blender poses are not automatically ground truth for generated frames. A video
generator may preserve the impression of an orbit while changing its speed,
phase, direction, radius, elevation, focal length, framing, or endpoint. It may
also deform the subject, in which case no single rigid camera trajectory can
fully explain the output.

## Required artifacts

Preserve the following under immutable identities:

- Blender scene and version;
- render resolution, frame rate, frame range, and pixel aspect;
- camera `matrix_world` for every blockout frame;
- focal length, sensor dimensions and fit, lens shift, clipping settings, and
  render dimensions;
- blockout video manifest;
- generated video and extracted-frame manifest;
- generation model, checkpoint or endpoint identity, seed, prompt, duration,
  and conditioning configuration;
- COLMAP version, database, sparse model, and preparation report.

The generated frames and Blender frames should initially have the same count
and timing so direct index correspondence can be evaluated before introducing
any temporal realignment.

## Coordinate and camera conventions

Blender camera and COLMAP camera coordinates differ:

```text
Blender local camera: X right, Y up, -Z forward
COLMAP local camera:  X right, Y down, Z forward
```

Blender `matrix_world` is camera-to-world. COLMAP stores a world-to-camera
rotation and translation. Conversion must:

1. flip the camera-local Y and Z directions;
2. invert camera-to-world into world-to-camera;
3. encode rotation with the COLMAP Hamilton quaternion convention;
4. compute translation in the COLMAP world-to-camera convention;
5. verify camera centers using `C = -R^T T`;
6. validate the conversion by rendering axes and known anchor views.

COLMAP's camera-coordinate and text-model conventions are documented at:
<https://colmap.github.io/format.html>.

Blender intrinsics must be derived from focal length, sensor width and height,
sensor fit, render dimensions, pixel aspect, and lens shift. Record
`fx`, `fy`, `cx`, and `cy` explicitly. Do not assume that the generated video
preserves the blockout focal length merely because the framing looks similar.

## Temporal correspondence

Equal frame count does not prove frame-to-frame camera correspondence. Evaluate
in increasing order of flexibility:

1. direct index alignment: Blender frame `i` to generated frame `i`;
2. circular phase offset for a closed orbit;
3. rotation direction or reflected traversal;
4. monotonic time alignment;
5. constrained Dynamic Time Warping only as a separate diagnostic.

Always report direct alignment before time-warped alignment. Otherwise temporal
realignment may hide the generator's failure to follow the requested timing.

Front, left profile, rear, and right profile anchors should resolve circular
phase and reflection ambiguity. A front anchor is especially important for a
near-circular, planar trajectory.

## Spatial trajectory alignment

COLMAP reconstruction has arbitrary scale, rotation, and translation. Align it
to Blender with a similarity transform:

```text
C_blender ~= scale * R_align * C_colmap + translation
```

Use Umeyama or an equivalent Sim(3) fit on corresponding camera centers. A
circular path can be under-constrained, so validate orientation, camera forward
vectors, traversal direction, and anchor views rather than relying only on
center error.

Preserve both the raw COLMAP trajectory and aligned result. Alignment is an
analysis transform, not evidence that Blender poses are correct for the output.

## Comparison metrics

### Camera position and path

- absolute trajectory error after Sim(3) alignment;
- center error per frame, normalized by orbit radius;
- orbit radius and radius drift;
- height and elevation error;
- closure gap for expected closed or half-orbit endpoints;
- deviation from the intended orbit plane;
- trajectory smoothness and positional jerk.

### Camera orientation

- geodesic rotation error in degrees;
- camera-forward or look-at direction error;
- roll error;
- orientation smoothness and angular jerk.

### Timing

- orbit angle by frame;
- angular velocity and acceleration;
- phase drift;
- error before and after constrained temporal alignment;
- whether profile and rear anchors occur at intended frames.

### Intrinsics and composition

- Blender focal length against COLMAP-estimated focal length;
- horizontal and vertical field of view;
- principal point;
- generated subject scale and center in frame;
- evidence of zoom or changing focal length.

Recommended evidence includes aligned top and side trajectory plots, camera
frustums, angle/radius/elevation curves, per-frame rotation error, and a frame
strip at the semantic anchor angles.

## Interpreting disagreement

Trajectory difference has at least three possible causes:

```text
the generator changed the camera
COLMAP estimated the camera incorrectly
the generator changed subject geometry
```

Do not label Blender/COLMAP disagreement as camera error without additional
evidence. Compare it with:

- COLMAP reprojection error and track support;
- optical-flow and longer-range feature consistency;
- depth consistency;
- local identity and silhouette drift;
- render-back residuals from reconstructed cameras.

Example interpretations:

```text
large trajectory difference + strong tracks + stable subject
  -> generator likely changed the camera

large trajectory difference + weak tracks or high reprojection error
  -> COLMAP estimate may be unreliable

smooth camera trajectory + local flow/depth contradiction
  -> subject morphing is a stronger explanation
```

## Camera-use candidates

### A. COLMAP cameras

Current baseline:

```text
generated frames -> COLMAP cameras -> static 3DGS
```

This estimates the camera that best explains the output images under a rigid
scene model, subject to feature and geometry quality.

### B. Fixed Blender cameras

Convert and use Blender poses without COLMAP pose estimation:

```text
generated frames + aligned Blender cameras -> static 3DGS
```

This is initially a diagnostic. It may fail if the generator changed timing or
camera motion. To isolate extrinsics, first compare with the same accepted
intrinsics in both runs; test Blender versus COLMAP intrinsics separately.

### C. Blender initialization followed by refinement

Initialize cameras from Blender, triangulate from image correspondences, and
allow bundle adjustment to refine poses. This may help when the white
background or weak texture makes unconstrained initialization difficult.

Record pose changes from initialization to final solution and do not describe
the result as fixed Blender cameras.

### D. Blender as a soft pose prior

Optimize reprojection while penalizing departure from the intended camera:

```text
loss = reprojection_error
     + lambda_position * position_prior_error
     + lambda_rotation * rotation_prior_error
```

Recent COLMAP releases include `pose_prior_mapper`, with configurable position
uncertainty. Official overview:
<https://github.com/colmap/colmap/blob/master/doc/faq.rst>.

The currently deployed repository image uses COLMAP 3.9.1 and must not be
silently upgraded. Pose-prior reconstruction therefore requires a separately
versioned runtime and validation. COLMAP's built-in pose-prior path primarily
targets position priors; full Blender orientation constraints may require a
custom refinement stage.

### E. Post-COLMAP hybrid trajectory

After Sim(3) and temporal alignment, compute per-frame correction:

```text
delta_i = inverse(Blender_i) * COLMAP_i
```

Filter implausible or high-frequency corrections and form:

```text
Hybrid_i = Blender_i * smooth(delta_i)
```

This lets Blender define the orbit's global shape while COLMAP supplies smooth
frame-specific corrections. It is a custom research method, not an existing
Nerfstudio option. Preserve the exact filter, window, thresholds, and raw
corrections.

## Proposed experiment order

### Phase 1: camera adherence only

Do not train a splat first. Produce:

| Identity | Trajectory | Purpose |
|---|---|---|
| C0 | raw Blender | intended camera motion |
| C1 | raw COLMAP | output-derived motion |
| C2 | COLMAP aligned to Blender with Sim(3) | spatial difference |
| C3 | spatially and temporally aligned COLMAP | separate timing drift from path drift |

Acceptance for continuing is not a low trajectory error alone. Camera
correspondence, coordinate conversion, semantic anchors, and reprojection
evidence must all be plausible.

### Phase 2: controlled 3DGS comparison

Freeze generated RGB, frame subset, intrinsics, trainer settings, step count,
background, and evaluation cameras. Change only extrinsics:

| Run | Extrinsics | Purpose |
|---|---|---|
| GS-A | COLMAP | baseline |
| GS-B | aligned Blender, fixed | test direct adherence |
| GS-C | Blender plus smoothed COLMAP corrections | test hybrid prior |

Compare training diagnostics, matched render paths, edge stability, floaters,
Gaussian count, and held-out metrics if a frozen split exists. A failed fixed
Blender run remains useful evidence that the generator did not preserve exact
camera control.

### Phase 3: pose-prior reconstruction

Only after Phase 1 and Phase 2, evaluate Blender initialization or soft priors
under a separately pinned modern COLMAP runtime. Freeze prior covariance and
refinement settings before the main run.

## Main hypotheses

Test these separately:

1. the generator preserves orbit shape but changes temporal parameterization;
2. it preserves position but changes focal length or framing;
3. COLMAP has local pose noise that a smooth Blender prior can reduce;
4. remaining disagreement is caused by non-rigid subject drift and cannot be
   corrected by camera fusion alone.

If the first hypothesis holds, phase/time alignment may make Blender cameras
useful. If radius, elevation, or focal length drift, soft or hybrid refinement
is required. If local subject geometry changes, proceed to inconsistency maps,
robust loss, or depth/feature confidence rather than forcing the camera to fit.

