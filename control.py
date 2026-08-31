"""Invoke functions from the deployed Modal App without a web endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

from splat_job import load_config


APP_NAME = "gaussian-splat-trainer"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("doctor", "prepare", "train", "export", "status"),
    )
    parser.add_argument("--kind", default="smoke", choices=("smoke", "main"))
    parser.add_argument("--config", type=Path, default=Path("configs/mix_back_clean_v1.json"))
    parser.add_argument("--environment", default="main")
    args = parser.parse_args()

    config = load_config(args.config)
    function = modal.Function.from_name(
        APP_NAME,
        args.stage,
        environment_name=args.environment,
    )
    if args.stage in {"train", "export"}:
        result = function.remote(config, args.kind)
    elif args.stage in {"doctor", "prepare"}:
        result = function.remote(config)
    else:
        result = function.remote(config["job_id"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
