import tempfile
import unittest
from pathlib import Path

from experiments.dinov3_grid_unfreeze2.config import load_config


class GridUnfreezeTwoConfigTests(unittest.TestCase):
    def test_minimal_config_has_safe_finetuning_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text('[data]\ndataset_dir = "/dataset"\n', encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.model.unfreeze_last_n_blocks, 2)
        self.assertTrue(config.model.unfreeze_final_norm)
        self.assertEqual(config.training.eval_every, 1)
        self.assertEqual(config.training.batch_size, 8)
        self.assertEqual(config.training.gradient_accumulation_steps, 2)
        self.assertEqual(config.runtime.mixed_precision, "fp16")

    def test_invalid_mixed_precision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(
                '[data]\ndataset_dir = "/dataset"\n[runtime]\nmixed_precision = "fp8"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mixed_precision"):
                load_config(path)

    def test_unknown_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(
                '[data]\ndataset_dir = "/dataset"\n[typo]\nvalue = 1\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Unknown configuration section"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
