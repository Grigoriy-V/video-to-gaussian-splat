from pathlib import Path
import unittest

from splat_job import (
    colmap_conversion_command,
    expected_run_dir,
    export_command,
    load_config,
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


if __name__ == "__main__":
    unittest.main()
