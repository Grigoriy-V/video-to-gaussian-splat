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
    return [
        "ns-train",
        training["method"],
        "--data",
        str(root / "processed"),
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
        "--pipeline.model.background-color",
        training["background_color"],
        "nerfstudio-data",
        "--downscale-factor",
        str(training["downscale_factor"]),
        "--eval-mode",
        training["eval_mode"],
        "--load-3D-points",
        str(training["load_3d_points"]),
    ]


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
