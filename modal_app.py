"""Modal batch runner for COLMAP and Nerfstudio Splatfacto."""

from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import modal

from splat_job import (
    colmap_conversion_command,
    dataset_root,
    expected_run_dir,
    export_command,
    image_manifest_sha256,
    job_root,
    milestone_export_command,
    prepare_command,
    run_id,
    checkpoint_steps,
    resolve_milestone_checkpoint,
    training_command,
    validate_config,
)


APP_NAME = "gaussian-splat-trainer"
VOLUME_NAME = "gaussian-splat-runs"
VOLUME_ROOT = Path("/runs")
NERFSTUDIO_IMAGE = "ghcr.io/nerfstudio-project/nerfstudio:1.1.5"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry(NERFSTUDIO_IMAGE)
    .entrypoint([])
    .add_local_python_source("splat_job")
)
control_image = modal.Image.debian_slim().add_local_python_source("splat_job")

gpu_options = {
    "image": image,
    "gpu": "L4",
    "cpu": 8,
    "memory": 32768,
    "volumes": {str(VOLUME_ROOT): volume},
    "min_containers": 0,
    "buffer_containers": 0,
    "max_containers": 1,
    "scaledown_window": 2,
    "retries": 0,
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; see {log_path}"
        )
    return elapsed


def _capture(command: list[str]) -> str:
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    return completed.stdout.strip()


def _checkpoint_config(source: Path, target: Path, load_step: int) -> None:
    """Clone a Nerfstudio config while selecting one exact saved checkpoint."""
    text = source.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?m)^(load_step:\s*).*$",
        rf"\g<1>{load_step}",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"could not set load_step in {source}")
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(updated, encoding="utf-8")


def _validate_input(config: dict[str, Any], input_dir: Path) -> dict[str, Any]:
    digest, images = image_manifest_sha256(input_dir)
    source = config["source"]
    if len(images) != source["expected_image_count"]:
        raise RuntimeError(f"expected {source['expected_image_count']} PNGs, found {len(images)}")
    if digest != source["sequence_manifest_sha256"]:
        raise RuntimeError("input sequence manifest SHA-256 does not match the config")

    dimensions: set[tuple[int, int]] = set()
    for path in images:
        with path.open("rb") as frame:
            header = frame.read(24)
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise RuntimeError(f"invalid PNG header: {path.name}")
        dimensions.add(struct.unpack(">II", header[16:24]))
    expected = (source["expected_width"], source["expected_height"])
    if dimensions != {expected}:
        raise RuntimeError(f"expected only {expected} images, found {sorted(dimensions)}")
    return {"image_count": len(images), "dimensions": list(expected), "manifest_sha256": digest}


def _png_color_type(path: Path) -> int:
    with path.open("rb") as frame:
        header = frame.read(26)
    if len(header) != 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"invalid PNG header: {path.name}")
    return header[25]


def _colmap_model_image_count(model_dir: Path) -> int:
    images_bin = model_dir / "images.bin"
    with images_bin.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise RuntimeError(f"invalid COLMAP images.bin: {images_bin}")
    return int(struct.unpack("<Q", header)[0])


@app.function(**gpu_options, timeout=5 * 60)
def doctor(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    root = job_root(VOLUME_ROOT, config["job_id"])
    report = {
        "status": "passed",
        "gpu": _capture([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]),
        "runtime": _capture([
            "python",
            "-c",
            "import torch, gsplat; "
            "print(f'torch={torch.__version__} cuda={torch.version.cuda} "
            "available={torch.cuda.is_available()} gsplat={gsplat.__version__} "
            "capability={torch.cuda.get_device_capability()}')",
        ]),
        "colmap": _capture(["colmap", "-h"]).splitlines()[0],
        "nerfstudio": _capture(["ns-train", "--help"]).splitlines()[0],
        "image": NERFSTUDIO_IMAGE,
    }
    _write_json(root / "doctor_report.json", report)
    volume.commit()
    return report


@app.function(**gpu_options, timeout=45 * 60)
def prepare(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    root = job_root(VOLUME_ROOT, config["job_id"])
    if (root / "processed").exists():
        raise RuntimeError(f"refusing to overwrite existing preparation at {root / 'processed'}")

    input_report = _validate_input(config, root / "input")
    _write_json(root / "job_config.json", config)
    command = prepare_command(config, root)
    elapsed = _run(command, root / "logs" / "prepare.log")

    sparse_root = root / "processed" / "colmap" / "sparse"
    model_counts = {
        path.name: _colmap_model_image_count(path)
        for path in sorted(sparse_root.iterdir())
        if path.is_dir() and (path / "images.bin").exists()
    }
    if not model_counts:
        raise RuntimeError(f"COLMAP produced no sparse models under {sparse_root}")
    selected_model = "0"
    conversion = None
    conversion_elapsed = 0.0
    if config["prepare"]["select_largest_colmap_model"]:
        selected_model = max(model_counts, key=lambda name: (model_counts[name], -int(name)))
        if selected_model != "0":
            conversion = colmap_conversion_command(root, selected_model)
            conversion_elapsed = _run(
                conversion,
                root / "logs" / f"prepare-select-model-{selected_model}.log",
            )

    transforms_path = root / "processed" / "transforms.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    registered = len(transforms.get("frames", []))
    minimum = config["prepare"]["minimum_registered_images"]
    report = {
        "status": "accepted" if registered >= minimum else "rejected",
        "registered_images": registered,
        "minimum_registered_images": minimum,
        "elapsed_seconds": elapsed + conversion_elapsed,
        "input": input_report,
        "command": command,
        "colmap_models": model_counts,
        "selected_colmap_model": selected_model,
        "conversion_command": conversion,
    }
    _write_json(root / "prepare_report.json", report)
    volume.commit()
    if registered < minimum:
        raise RuntimeError(f"camera solve registered only {registered}/{input_report['image_count']} images")
    return report


@app.function(
    image=control_image,
    volumes={str(VOLUME_ROOT): volume},
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    timeout=5 * 60,
)
def attach_masks(config: dict[str, Any]) -> dict[str, Any]:
    """Create an immutable masked Nerfstudio dataset from an accepted camera solve."""
    validate_config(config)
    masks = config.get("masks", {"enabled": False})
    if not masks["enabled"]:
        raise RuntimeError("masks.enabled must be true")

    root = job_root(VOLUME_ROOT, config["job_id"])
    prepare_report_path = root / "prepare_report.json"
    if not prepare_report_path.exists():
        raise RuntimeError("prepare must complete before masks can be attached")
    prepare_report = json.loads(prepare_report_path.read_text(encoding="utf-8"))
    if prepare_report.get("status") != "accepted":
        raise RuntimeError("camera solve did not pass the registration gate")

    target = dataset_root(config, root)
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing masked dataset at {target}")

    mask_source = root / "masks" / masks["mask_id"]
    digest, mask_paths = image_manifest_sha256(mask_source)
    if len(mask_paths) != masks["expected_count"]:
        raise RuntimeError(f"expected {masks['expected_count']} masks, found {len(mask_paths)}")
    if digest != masks["sequence_manifest_sha256"]:
        raise RuntimeError("mask sequence manifest SHA-256 does not match the config")

    expected_dimensions = (config["source"]["expected_width"], config["source"]["expected_height"])
    for mask_path in mask_paths:
        with mask_path.open("rb") as frame:
            header = frame.read(24)
        dimensions = struct.unpack(">II", header[16:24]) if len(header) == 24 else None
        if dimensions != expected_dimensions:
            raise RuntimeError(f"mask {mask_path.name} has dimensions {dimensions}, expected {expected_dimensions}")
        if _png_color_type(mask_path) != 0:
            raise RuntimeError(f"mask {mask_path.name} must be a one-channel grayscale PNG")

    transforms_path = root / "processed" / "transforms.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = transforms.get("frames", [])
    if len(frames) != len(mask_paths):
        raise RuntimeError(f"camera solve has {len(frames)} frames but mask set has {len(mask_paths)}")

    target.mkdir(parents=True)
    target_masks = target / "masks"
    shutil.copytree(mask_source, target_masks)
    for frame, mask_path in zip(frames, mask_paths, strict=True):
        image_path = Path(frame["file_path"])
        if not image_path.is_absolute():
            image_path = (root / "processed" / image_path).resolve()
        frame["file_path"] = str(image_path)
        frame["mask_path"] = str((target_masks / mask_path.name).resolve())

    ply_path = transforms.get("ply_file_path")
    if ply_path and not Path(ply_path).is_absolute():
        transforms["ply_file_path"] = str((root / "processed" / ply_path).resolve())
    _write_json(target / "transforms.json", transforms)

    report = {
        "status": "completed",
        "mask_id": masks["mask_id"],
        "mask_count": len(mask_paths),
        "mask_manifest_sha256": digest,
        "dataset_path": str(target),
        "camera_source": str(root / "processed"),
    }
    _write_json(root / f"attach_masks_{masks['mask_id']}_report.json", report)
    volume.commit()
    return report


@app.function(
    image=control_image,
    volumes={str(VOLUME_ROOT): volume},
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    timeout=5 * 60,
)
def attach_rgba(config: dict[str, Any]) -> dict[str, Any]:
    """Attach RGBA images to an accepted camera solve without rerunning COLMAP."""
    validate_config(config)
    rgba = config.get("rgba", {"enabled": False})
    if not rgba["enabled"]:
        raise RuntimeError("rgba.enabled must be true")

    root = job_root(VOLUME_ROOT, config["job_id"])
    camera_root = job_root(VOLUME_ROOT, rgba["camera_source_job_id"])
    camera_report_path = camera_root / "prepare_report.json"
    if not camera_report_path.exists():
        raise RuntimeError("camera source has no prepare report")
    camera_report = json.loads(camera_report_path.read_text(encoding="utf-8"))
    if camera_report.get("status") != "accepted":
        raise RuntimeError("camera source preparation was not accepted")

    input_report = _validate_input(config, root / "input")
    _, images = image_manifest_sha256(root / "input")
    non_rgba = [path.name for path in images if _png_color_type(path) != 6]
    if non_rgba:
        raise RuntimeError(f"RGBA input contains non-RGBA PNG files: {non_rgba[:5]}")

    target = dataset_root(config, root)
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing RGBA dataset at {target}")

    source_transforms_path = camera_root / "processed" / "transforms.json"
    transforms = json.loads(source_transforms_path.read_text(encoding="utf-8"))
    frames = transforms.get("frames", [])
    if len(frames) != len(images):
        raise RuntimeError(
            f"camera source has {len(frames)} frames but RGBA input has {len(images)}"
        )

    target.mkdir(parents=True)
    target_images = target / "images"
    shutil.copytree(root / "input", target_images)
    for frame in frames:
        match = re.search(r"frame_(\d+)\.png$", str(frame.get("file_path", "")))
        if match is None:
            raise RuntimeError(f"cannot map camera frame path: {frame.get('file_path')}")
        source_index = int(match.group(1)) - 1
        if not 0 <= source_index < len(images):
            raise RuntimeError(f"camera frame index is outside RGBA input: {source_index}")
        frame["file_path"] = str((target_images / images[source_index].name).resolve())
        frame.pop("mask_path", None)

    ply_path = transforms.get("ply_file_path")
    if ply_path and not Path(ply_path).is_absolute():
        transforms["ply_file_path"] = str((camera_root / "processed" / ply_path).resolve())
    _write_json(target / "transforms.json", transforms)
    _write_json(root / "job_config.json", config)

    report = {
        "status": "accepted",
        "dataset_id": rgba["dataset_id"],
        "dataset_path": str(target),
        "camera_source_job_id": rgba["camera_source_job_id"],
        "registered_images": len(frames),
        "input": input_report,
        "png_color_type": 6,
    }
    _write_json(root / f"attach_rgba_{rgba['dataset_id']}_report.json", report)
    volume.commit()
    return report


@app.function(**gpu_options, timeout=2 * 60 * 60)
def train(config: dict[str, Any], kind: Literal["smoke", "main"]) -> dict[str, Any]:
    validate_config(config)
    root = job_root(VOLUME_ROOT, config["job_id"])
    rgba = config.get("rgba", {"enabled": False})
    if rgba["enabled"]:
        rgba_report = root / f"attach_rgba_{rgba['dataset_id']}_report.json"
        if not rgba_report.exists() or not dataset_root(config, root).exists():
            raise RuntimeError("attach_rgba must complete before RGBA training")
    else:
        prepare_report_path = root / "prepare_report.json"
        if not prepare_report_path.exists():
            raise RuntimeError("prepare must complete before training")
        prepare_report = json.loads(prepare_report_path.read_text(encoding="utf-8"))
        if prepare_report.get("status") != "accepted":
            raise RuntimeError("camera solve did not pass the registration gate")
    if config.get("masks", {"enabled": False})["enabled"] and not dataset_root(config, root).exists():
        raise RuntimeError("attach_masks must complete before masked training")

    target = expected_run_dir(config, root, kind)
    if target.exists():
        raise RuntimeError(
            f"refusing to overwrite or resume existing run at {target}; "
            "Nerfstudio 1.1.5 Splatfacto resume is not trusted because a "
            "verified continuation advanced steps without updating tensors"
        )
    command = training_command(config, root, kind)
    identity = run_id(config, kind)
    elapsed = _run(command, root / "logs" / f"train-{identity}.log")
    if not (target / "config.yml").exists():
        raise RuntimeError(f"training completed but {target / 'config.yml'} is missing")

    exports = []
    export_elapsed = 0.0
    if kind == "main":
        for iterations in config["training"].get("export_milestones", []):
            checkpoint_step, checkpoint = resolve_milestone_checkpoint(
                target / "nerfstudio_models", iterations
            )
            milestone_name = f"{iterations // 1000}k"
            legacy_export = root / "export" / identity / "splat.ply"
            if iterations < config["training"]["main_steps"] and legacy_export.exists():
                exports.append({
                    "iterations": iterations,
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_path": str(checkpoint),
                    "elapsed_seconds": 0.0,
                    "reused_existing": True,
                    "output_dir": str(legacy_export.parent),
                })
                continue
            milestone_root = root / "export" / f"{identity}-{milestone_name}"
            completed_ply = milestone_root / "splat" / "splat.ply"
            if completed_ply.exists():
                exports.append({
                    "iterations": iterations,
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_path": str(checkpoint),
                    "elapsed_seconds": 0.0,
                    "reused_existing": True,
                    "output_dir": str(completed_ply.parent),
                })
                continue
            if milestone_root.exists():
                raise RuntimeError(f"refusing to overwrite milestone export at {milestone_root}")
            selected_config = milestone_root / "config.yml"
            _checkpoint_config(target / "config.yml", selected_config, checkpoint_step)
            export_dir = milestone_root / "splat"
            export_cmd = milestone_export_command(selected_config, export_dir)
            duration = _run(
                export_cmd,
                root / "logs" / f"export-{identity}-{milestone_name}.log",
            )
            export_elapsed += duration
            exports.append({
                "iterations": iterations,
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": str(checkpoint),
                "elapsed_seconds": duration,
                "command": export_cmd,
                "output_dir": str(export_dir),
            })

    report = {
        "status": "completed",
        "kind": kind,
        "run_id": run_id(config, kind),
        "elapsed_seconds": elapsed,
        "export_elapsed_seconds": export_elapsed,
        "milestone_exports": exports,
        "command": command,
        "config_path": str(target / "config.yml"),
    }
    _write_json(root / f"train_{identity}_report.json", report)
    volume.commit()
    return report


@app.function(**gpu_options, timeout=30 * 60)
def export(config: dict[str, Any], kind: Literal["smoke", "main"]) -> dict[str, Any]:
    validate_config(config)
    root = job_root(VOLUME_ROOT, config["job_id"])
    target = root / "export" / run_id(config, kind)
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing export at {target}")
    source_config = expected_run_dir(config, root, kind) / "config.yml"
    if not source_config.exists():
        raise RuntimeError(f"missing trained config: {source_config}")

    command = export_command(config, root, kind)
    identity = run_id(config, kind)
    elapsed = _run(command, root / "logs" / f"export-{identity}.log")
    report = {
        "status": "completed",
        "kind": kind,
        "run_id": run_id(config, kind),
        "elapsed_seconds": elapsed,
        "command": command,
        "output_dir": str(target),
    }
    _write_json(root / f"export_{identity}_report.json", report)
    volume.commit()
    return report


@app.function(
    image=control_image,
    volumes={str(VOLUME_ROOT): volume},
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    scaledown_window=2,
    retries=0,
    timeout=60,
)
def status(job_id: str) -> dict[str, Any]:
    root = job_root(VOLUME_ROOT, job_id)
    reports = {}
    for path in sorted(root.glob("*_report.json")):
        reports[path.name] = json.loads(path.read_text(encoding="utf-8"))
    return {"job_id": job_id, "exists": root.exists(), "reports": reports}
