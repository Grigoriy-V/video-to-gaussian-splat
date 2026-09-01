# Experiment 04: soft-alpha RGBA background removal

**Status:** accepted

**Date:** 2026-09-01

## Objective

Remove the persistent white silhouette fringe and background splats without
repeating the quality regression caused by the earlier binary foreground mask.

## Input and configuration

```text
input: AI-generated character orbit
frames: 94 RGBA PNG
resolution: 768 x 960
alpha source: BiRefNet/VITMatte local preprocessing
alpha adjustment: inward adjustment 2, blur 1 px
separate mask files: none
camera source: accepted 94-camera native baseline solve
COLMAP rerun: no
method: Nerfstudio Splatfacto, default strategy
training background: random per iteration
GPU: Nvidia L4
steps: 30,000
```

The frozen RGBA sequence manifest is
`4e12367cfa629f27d1a11b0dc9e4a8d352333193f06f922ad256152cba92b9f6`.
All 94 inputs were verified as 768 x 960 RGBA PNG files and attached to the
accepted baseline camera solution without rerunning COLMAP.

## Runtime and artifact

| Item | Result |
|---|---:|
| Smoke training | 2,000 steps / 90.77 seconds |
| Main training | 30,000 steps / 751.74 seconds |
| Final export | 33.79 seconds |
| Exported Gaussians | 68,899 |
| PLY size | 17,088,544 bytes / 16.30 MiB |
| PLY SHA-256 | `3bac2cc437e7a85d3f446e0f46a46009c2dcd30d18d502a271b3865f7cff694d` |
| Local artifact | `results/experiment-04-soft-alpha-30k/splat-30k.ply` |

The Modal GPU scaled to zero after the export.

## Visual result

Human visual inspection accepted the result:

- the white silhouette fringe is gone;
- the previous floating background outliers are absent;
- the character remains coherent and usable;
- only a small number of splats remain below the cropped subject where the
  source sequence provides no observations.

The lower residual is an expected coverage limitation and is not material for
the intended bust reconstruction. The experiment does not claim recovery of
geometry absent from the source views.

## Decision

Adopt native-resolution soft RGBA with per-iteration random background as the
accepted foreground/background treatment. Do not return to the rejected binary
mask approach for this sequence. Preserve the x2 unmasked result as the detail
comparison, but use this experiment as the clean-background baseline for the
next reconstruction tests.

