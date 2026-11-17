import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

from tools.benchmark import aggregate, main
from tools.stage_data import stage


class Benchmarks(unittest.TestCase):
    def test_evaluation_uses_user_supplied_files(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "my-eval"
            folder.mkdir()
            (folder / "records.json").write_text('{"records":[]}')
            errors = StringIO()
            with patch("sys.argv", ["benchmark", str(folder)]), redirect_stderr(errors):
                with self.assertRaises(SystemExit):
                    main()
            self.assertIn(str(folder / "episodes.json"), errors.getvalue())

    def test_seed_summary_uses_sample_sd_in_points(self):
        rows = [dict(all_saved_masks_recounted=True, requests=10,
                     binding=dict(manifest_sha256="m",episodes_sha256="e",weights_sha256=str(i)),
                     pooled=dict(binary_mIoU=v,F1=v)) for i,v in enumerate((.4,.5,.6))]
        result = aggregate(rows)["binary_mIoU"]
        self.assertAlmostEqual(result["mean_percent"], 50)
        self.assertAlmostEqual(result["sample_sd_points"], 10)
        with self.assertRaises(ValueError):
            aggregate([rows[0],rows[0]])
        rows[1]["binding"]["episodes_sha256"] = "other"
        with self.assertRaises(ValueError):
            aggregate(rows)

    def test_stage_prepared_records(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = dict(id="pair", positive=["color"], negative=["size"])
            source, target = dict(base), dict(base)
            for key in ("before","after","mask"):
                (root/key).write_bytes(key.encode())
                source[key], target[key] = key, "assets/"+key
            a, b = root/"source.json", root/"protocol.json"
            a.write_text(json.dumps({"records":[source]}))
            b.write_text(json.dumps({"records":[target]}))
            for _ in range(2):
                self.assertEqual(stage(b, [a], root/"data"), 3)
            (root/"mask").write_bytes(b"new mask")
            with self.assertRaises(ValueError):
                stage(b, [a], root/"data")
