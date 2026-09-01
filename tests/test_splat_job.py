from pathlib import Path
import unittest

from splat_job import (
    colmap_conversion_command,
    dataset_root,
    expected_run_dir,
    export_command,
    load_config,
    milestone_checkpoint_step,
    milestone_export_command,
    prepare_command,
    training_command,
    validate_config,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "mix_back_clean_v1.json"


class SplatJobTests(unittest.TestCase):
    def test_config_is_valid(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(config["job_id"], "mix-back-clean-v1")
        self.assertEqual(config["training"]["strategy"], "default")
        self.assertEqual(config["source"]["expected_image_count"], 94)

    def test_prepare_command_uses_fixed_camera_and_exhaustive_matching(self) -> None:
        config = load_config(CONFIG_PATH)
        command = prepare_command(config, Path("/runs/jobs/mix-back-clean-v1"))
        self.assertEqual(command[:2], ["ns-process-data", "images"])
        self.assertEqual(command[command.index("--camera-type") + 1], "simple_pinhole")
        self.assertEqual(command[command.index("--matching-method") + 1], "exhaustive")
        self.assertEqual(command[-1], "--gpu")

    def test_training_runs_are_non_overlapping(self) -> None:
        config = load_config(CONFIG_PATH)
        root = Path("/runs/jobs/mix-back-clean-v1")
        smoke = expected_run_dir(config, root, "smoke")
        main = expected_run_dir(config, root, "main")
        self.assertNotEqual(smoke, main)
        self.assertIn("2000", training_command(config, root, "smoke"))
        self.assertIn("30000", training_command(config, root, "main"))
        self.assertEqual(
            training_command(config, root, "main")[
                training_command(config, root, "main").index("--pipeline.model.background-color") + 1
            ],
            "random",
        )

    def test_colmap_conversion_selects_an_explicit_model(self) -> None:
        root = Path("/runs/jobs/mix-back-clean-v1-largest-model")
        command = colmap_conversion_command(root, "1")
        self.assertIn("--skip-colmap", command)
        self.assertIn("--skip-image-processing", command)
        self.assertEqual(command[command.index("--colmap-model-path") + 1], "colmap/sparse/1")
        with self.assertRaisesRegex(ValueError, "numeric"):
            colmap_conversion_command(root, "../1")

    def test_export_loads_the_exact_run_config(self) -> None:
        config = load_config(CONFIG_PATH)
        root = Path("/runs/jobs/mix-back-clean-v1")
        command = export_command(config, root, "main")
        load_config_path = Path(command[command.index("--load-config") + 1])
        self.assertEqual(load_config_path, expected_run_dir(config, root, "main") / "config.yml")

    def test_mcmc_is_rejected_in_first_experiment(self) -> None:
        config = load_config(CONFIG_PATH)
        config["training"]["strategy"] = "mcmc"
        with self.assertRaisesRegex(ValueError, "excludes MCMC"):
            validate_config(config)

    def test_masked_dataset_has_separate_identity(self) -> None:
        config = load_config(CONFIG_PATH)
        config["masks"] = {
            "enabled": True,
            "mask_id": "birefnet-binary-v1",
            "expected_count": 94,
            "sequence_manifest_sha256": "a" * 64,
        }
        validate_config(config)
        root = Path("/runs/jobs/mix-back-clean-v1")
        self.assertEqual(dataset_root(config, root), root / "datasets" / "birefnet-binary-v1")
        self.assertEqual(
            training_command(config, root, "smoke")[training_command(config, root, "smoke").index("--data") + 1],
            str(root / "datasets" / "birefnet-binary-v1"),
        )

    def test_mask_count_must_match_rgb_count(self) -> None:
        config = load_config(CONFIG_PATH)
        config["masks"] = {
            "enabled": True,
            "mask_id": "birefnet-binary-v1",
            "expected_count": 93,
            "sequence_manifest_sha256": "a" * 64,
        }
        with self.assertRaisesRegex(ValueError, "must equal"):
            validate_config(config)

    def test_milestone_training_keeps_checkpoints(self) -> None:
        config = load_config(CONFIG_PATH)
        config["training"].update({
            "main_steps": 40000,
            "steps_per_save": 10000,
            "export_milestones": [30000, 40000],
        })
        validate_config(config)
        command = training_command(config, Path("/runs/jobs/test"), "main")
        self.assertEqual(command[command.index("--steps-per-save") + 1], "10000")
        self.assertEqual(
            command[command.index("--save-only-latest-checkpoint") + 1], "False"
        )
        self.assertEqual(milestone_checkpoint_step(30000), 29999)
        export = milestone_export_command(Path("selected.yml"), Path("export-30k"))
        self.assertEqual(export[export.index("--load-config") + 1], "selected.yml")

    def test_milestones_must_end_at_main_steps(self) -> None:
        config = load_config(CONFIG_PATH)
        config["training"].update({
            "main_steps": 40000,
            "steps_per_save": 10000,
            "export_milestones": [30000],
        })
        with self.assertRaisesRegex(ValueError, "final export milestone"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
