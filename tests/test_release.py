import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.check_assets import check
from vecd.cli import main


class ReleaseContracts(unittest.TestCase):
    def test_four_image_cli_requires_no_cues(self):
        args = ["infer", "--example-before", "a.png", "--example-after", "b.png",
                "--query-before", "c.png", "--query-after", "d.png", "--weights", "w.safetensors",
                "--model-config", "config.json", "--output", "outputs/demo", "--device", "cpu"]
        with patch("vecd.cli.infer") as run:
            main(args)
        parsed = run.call_args.args[0]
        self.assertEqual(parsed.example_before, "a.png")
        self.assertIsNone(parsed.point)
        self.assertIsNone(parsed.example_mask)

    def test_missing_and_corrupt_weights_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = b"test fixture, not model weights"
            manifest = {"seeds": {"1": {"file": "model.safetensors", "bytes": len(good),
                                        "sha256": hashlib.sha256(good).hexdigest()}}}
            self.assertFalse(check(manifest, root)["complete"])
            (root / "model.safetensors").write_bytes(good)
            self.assertTrue(check(manifest, root)["complete"])
            (root / "model.safetensors").write_bytes(b"x" * len(good))
            self.assertEqual(check(manifest, root)["files"][0]["status"], "sha256_mismatch")

    def test_inference_configs_are_bound_to_mainline_weights(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "weight_manifest.json").read_text())
        for seed, spec in manifest["seeds"].items():
            cfg = json.loads((root / f"configs/inference/seed{seed}.json").read_text())
            self.assertEqual(cfg["model_sha256"], spec["sha256"])
            self.assertEqual(cfg["thresholds"], spec["thresholds"])
            self.assertEqual(cfg["main_inference_mode"], "set")
            self.assertTrue(cfg["source_calibration_only"])


if __name__ == "__main__":
    unittest.main()
