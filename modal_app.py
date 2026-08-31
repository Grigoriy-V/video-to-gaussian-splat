"""Modal batch runner for COLMAP and Nerfstudio Splatfacto."""

from __future__ import annotations

import json
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import modal

from splat_job import (
    colmap_conversion_command,
    expected_run_dir,
    export_command,
    image_manifest_sha256,
    job_root,
    prepare_command,
    run_id,
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


@app.function(**gpu_options, timeout=2 * 60 * 60)
def train(config: dict[str, Any], kind: Literal["smoke", "main"]) -> dict[str, Any]:
    validate_config(config)
    root = job_root(VOLUME_ROOT, config["job_id"])
    prepare_report_path = root / "prepare_report.json"
    if not prepare_report_path.exists():
        raise RuntimeError("prepare must complete before training")
    prepare_report = json.loads(prepare_report_path.read_text(encoding="utf-8"))
    if prepare_report.get("status") != "accepted":
        raise RuntimeError("camera solve did not pass the registration gate")

    target = expected_run_dir(config, root, kind)
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing run at {target}")
    command = training_command(config, root, kind)
    elapsed = _run(command, root / "logs" / f"train-{kind}.log")
    if not (target / "config.yml").exists():
        raise RuntimeError(f"training completed but {target / 'config.yml'} is missing")

    report = {
        "status": "completed",
        "kind": kind,
        "run_id": run_id(config, kind),
        "elapsed_seconds": elapsed,
        "command": command,
        "config_path": str(target / "config.yml"),
    }
    _write_json(root / f"train_{kind}_report.json", report)
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
    elapsed = _run(command, root / "logs" / f"export-{kind}.log")
    report = {
        "status": "completed",
        "kind": kind,
        "run_id": run_id(config, kind),
        "elapsed_seconds": elapsed,
        "command": command,
        "output_dir": str(target),
    }
    _write_json(root / f"export_{kind}_report.json", report)
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
