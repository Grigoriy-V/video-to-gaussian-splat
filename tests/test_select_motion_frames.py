import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "select_motion_frames.py"
SPEC = importlib.util.spec_from_file_location("select_motion_frames", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class BiasedPositionsTest(unittest.TestCase):
    def test_keeps_requested_count_and_endpoints(self):
        positions = MODULE.biased_positions(124, 50)
        self.assertEqual(len(positions), 50)
        self.assertEqual(positions[0], 0)
        self.assertEqual(positions[-1], 123)
        self.assertEqual(positions, sorted(set(positions)))

    def test_power_concentrates_choices_at_the_middle(self):
        positions = MODULE.biased_positions(124, 50, power=3)
        gaps = [right - left for left, right in zip(positions, positions[1:])]
        self.assertGreater(gaps[0], gaps[len(gaps) // 2])
        self.assertGreater(gaps[-1], gaps[len(gaps) // 2])

    def test_center_moves_dense_region(self):
        positions = MODULE.biased_positions(124, 50, power=3, center=0.65)
        self.assertAlmostEqual(positions[24] / 123, 0.65, delta=0.03)

    def test_linear_mix_relaxes_the_central_cluster(self):
        curved = MODULE.biased_positions(124, 50, power=2.5, linear_mix=0)
        relaxed = MODULE.biased_positions(124, 50, power=2.5, linear_mix=0.35)
        curved_ones = sum(right - left == 1 for left, right in zip(curved, curved[1:]))
        relaxed_ones = sum(
            right - left == 1 for left, right in zip(relaxed, relaxed[1:])
        )
        self.assertLess(relaxed_ones, curved_ones)

    def test_select_copies_frames_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            output = root / "selected"
            source.mkdir()
            for index in range(10):
                (source / f"frame_{index:04d}.png").write_bytes(b"png")

            chosen = MODULE.select(source, output, 4, 2.0, 0.5, 0.2, False)

            self.assertEqual(len(chosen), 4)
            self.assertEqual(len(list(output.glob("*.png"))), 4)
            manifest = json.loads((output / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_frame_count"], 10)
            self.assertEqual(manifest["kept_frame_count"], 4)
            self.assertEqual(manifest["selected"][0]["source_index"], 1)
            self.assertEqual(manifest["selected"][-1]["source_index"], 10)


if __name__ == "__main__":
    unittest.main()
