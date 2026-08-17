import tempfile
import unittest
from pathlib import Path

from experiments.dinov3_regression.config import load_config


class ConfigTests(unittest.TestCase):
    def test_minimal_config_uses_notebook_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text('[data]\ndataset_dir = "/dataset"\n', encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.data.split_seed, 42)
        self.assertEqual(config.training.epochs, 30)
        self.assertEqual(config.model.head_hidden_dim, 256)

    def test_unknown_option_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(
                '[data]\ndataset_dir = "/dataset"\nunknown = true\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Unknown .*option"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
