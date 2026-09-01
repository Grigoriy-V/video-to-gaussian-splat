"""Pure helpers for validating jobs and building Nerfstudio commands."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
RUN_KINDS = {"smoke", "main"}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    job_id = config.get("job_id")
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise ValueError("job_id must contain only lowercase letters, digits, and hyphens")

    source = config.get("source", {})
    for key in ("expected_image_count", "expected_width", "expected_height"):
        if not isinstance(source.get(key), int) or source[key] <= 0:
            raise ValueError(f"source.{key} must be a positive integer")
    digest = source.get("sequence_manifest_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("source.sequence_manifest_sha256 must be a lowercase SHA-256")

    masks = config.get("masks", {"enabled": False})
    if not isinstance(masks.get("enabled"), bool):
        raise ValueError("masks.enabled must be a boolean")
    if masks["enabled"]:
        mask_id = masks.get("mask_id")
        if not isinstance(mask_id, str) or not JOB_ID_RE.fullmatch(mask_id):
            raise ValueError("masks.mask_id must contain only lowercase letters, digits, and hyphens")
        if masks.get("expected_count") != source["expected_image_count"]:
            raise ValueError("masks.expected_count must equal source.expected_image_count")
        mask_digest = masks.get("sequence_manifest_sha256")
        if not isinstance(mask_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", mask_digest):
            raise ValueError("masks.sequence_manifest_sha256 must be a lowercase SHA-256")

    rgba = config.get("rgba", {"enabled": False})
    if not isinstance(rgba.get("enabled"), bool):
        raise ValueError("rgba.enabled must be a boolean")
    if rgba["enabled"]:
        if masks["enabled"]:
            raise ValueError("rgba input and separate masks are mutually exclusive")
        dataset_id = rgba.get("dataset_id")
        if not isinstance(dataset_id, str) or not JOB_ID_RE.fullmatch(dataset_id):
            raise ValueError("rgba.dataset_id must contain only lowercase letters, digits, and hyphens")
        camera_job_id = rgba.get("camera_source_job_id")
        if not isinstance(camera_job_id, str) or not JOB_ID_RE.fullmatch(camera_job_id):
            raise ValueError("rgba.camera_source_job_id must be a safe job ID")

    prepare = config.get("prepare", {})
    minimum = prepare.get("minimum_registered_images")
    if not isinstance(minimum, int) or not 1 <= minimum <= source["expected_image_count"]:
        raise ValueError("prepare.minimum_registered_images is outside the image count")
    if not isinstance(prepare.get("select_largest_colmap_model"), bool):
        raise ValueError("prepare.select_largest_colmap_model must be a boolean")

    training = config.get("training", {})
    if training.get("method") != "splatfacto":
        raise ValueError("the first experiment supports only splatfacto")
    if training.get("strategy") != "default":
        raise ValueError("the first experiment intentionally excludes MCMC")
    if training.get("background_color") not in {"random", "white"}:
        raise ValueError("training.background_color must be random or white")
    for key in ("smoke_steps", "main_steps"):
        if not isinstance(training.get(key), int) or training[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    if training["smoke_steps"] >= training["main_steps"]:
        raise ValueError("smoke_steps must be lower than main_steps")
    milestones = training.get("export_milestones", [])
    if milestones:
        if not isinstance(milestones, list) or any(
            not isinstance(step, int) or step <= 0 for step in milestones
        ):
            raise ValueError("training.export_milestones must contain positive integers")
        if milestones != sorted(set(milestones)):
            raise ValueError("training.export_milestones must be sorted and unique")
        if milestones[-1] != training["main_steps"]:
            raise ValueError("the final export milestone must equal training.main_steps")
        save_interval = training.get("steps_per_save")
        if not isinstance(save_interval, int) or save_interval <= 0:
            raise ValueError("training.steps_per_save must be a positive integer")
        if any(step % save_interval for step in milestones):
            raise ValueError("each export milestone must align with training.steps_per_save")


def job_root(volume_root: Path, job_id: str) -> Path:
    if not JOB_ID_RE.fullmatch(job_id):
        raise ValueError("unsafe job_id")
    return volume_root / "jobs" / job_id


def image_manifest_sha256(image_dir: Path) -> tuple[str, list[Path]]:
    images = sorted(image_dir.glob("*.png"))
    lines = []
    for image in images:
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        lines.append(f"{image.name} {digest}\n")
    manifest = "".join(lines).encode("utf-8")
    return hashlib.sha256(manifest).hexdigest(), images


def dataset_root(config: dict[str, Any], root: Path) -> Path:
    rgba = config.get("rgba", {"enabled": False})
    if rgba["enabled"]:
        return root / "datasets" / rgba["dataset_id"]
    masks = config.get("masks", {"enabled": False})
    if masks["enabled"]:
        return root / "datasets" / masks["mask_id"]
    return root / "processed"


def prepare_command(config: dict[str, Any], root: Path) -> list[str]:
    prepare = config["prepare"]
    return [
        "ns-process-data",
        "images",
        "--data",
        str(root / "input"),
        "--output-dir",
        str(root / "processed"),
        "--camera-type",
        prepare["camera_type"],
        "--matching-method",
        prepare["matching_method"],
        "--sfm-tool",
        prepare["sfm_tool"],
        "--num-downscales",
        str(prepare["num_downscales"]),
        "--gpu",
    ]


def colmap_conversion_command(root: Path, model_name: str) -> list[str]:
    if not model_name.isdigit():
        raise ValueError("COLMAP model name must be numeric")
    return [
        "ns-process-data",
        "images",
        "--data",
        str(root / "processed" / "images"),
        "--output-dir",
        str(root / "processed"),
        "--skip-colmap",
        "--skip-image-processing",
        "--colmap-model-path",
        f"colmap/sparse/{model_name}",
        "--num-downscales",
        "0",
    ]


def run_id(config: dict[str, Any], kind: str) -> str:
    if kind not in RUN_KINDS:
        raise ValueError(f"run kind must be one of {sorted(RUN_KINDS)}")
    return config["training"][f"{kind}_run_id"]


def training_steps(config: dict[str, Any], kind: str) -> int:
    if kind not in RUN_KINDS:
        raise ValueError(f"run kind must be one of {sorted(RUN_KINDS)}")
    return int(config["training"][f"{kind}_steps"])


def expected_run_dir(config: dict[str, Any], root: Path, kind: str) -> Path:
    identity = run_id(config, kind)
    return root / "training" / identity / "splatfacto" / identity


def training_command(config: dict[str, Any], root: Path, kind: str) -> list[str]:
    training = config["training"]
    identity = run_id(config, kind)
    command = [
        "ns-train",
        training["method"],
        "--data",
        str(dataset_root(config, root)),
        "--output-dir",
        str(root / "training"),
        "--experiment-name",
        identity,
        "--timestamp",
        identity,
        "--vis",
        training["visualizer"],
        "--max-num-iterations",
        str(training_steps(config, kind)),
    ]
    if kind == "main" and training.get("export_milestones"):
        command.extend([
            "--steps-per-save",
            str(training["steps_per_save"]),
            "--save-only-latest-checkpoint",
            "False",
        ])
    command.extend([
        "--pipeline.model.background-color",
        training["background_color"],
        "nerfstudio-data",
        "--downscale-factor",
        str(training["downscale_factor"]),
        "--eval-mode",
        training["eval_mode"],
        "--load-3D-points",
        str(training["load_3d_points"]),
    ])
    return command


def export_command(config: dict[str, Any], root: Path, kind: str) -> list[str]:
    identity = run_id(config, kind)
    return [
        "ns-export",
        "gaussian-splat",
        "--load-config",
        str(expected_run_dir(config, root, kind) / "config.yml"),
        "--output-dir",
        str(root / "export" / identity),
    ]


def checkpoint_steps(checkpoint_dir: Path) -> list[int]:
    steps = []
    for path in checkpoint_dir.glob("step-*.ckpt"):
        match = re.fullmatch(r"step-(\d{9})\.ckpt", path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def resolve_milestone_checkpoint(
    checkpoint_dir: Path, iterations: int
) -> tuple[int, Path]:
    """Resolve periodic (N) or final (N-1) Nerfstudio checkpoint naming."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    available = set(checkpoint_steps(checkpoint_dir))
    for step in (iterations, iterations - 1):
        if step in available:
            return step, checkpoint_dir / f"step-{step:09d}.ckpt"
    raise FileNotFoundError(
        f"no checkpoint for {iterations} iterations under {checkpoint_dir}; "
        f"available steps: {sorted(available)}"
    )


def milestone_export_command(
    config_path: Path, output_dir: Path
) -> list[str]:
    return [
        "ns-export",
        "gaussian-splat",
        "--load-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
