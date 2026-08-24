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
        self.assertTrue(config.data.normalize_targets)

    def test_raw_target_comparison_config_disables_normalization(self):
        raw_config = load_config(
            "experiments/dinov3_grid_unfreeze2/config_raw_targets.toml"
        )
        normalized_config = load_config(
            "experiments/dinov3_grid_unfreeze2/config.toml"
        )

        self.assertFalse(raw_config.data.normalize_targets)
        self.assertEqual(
            raw_config.output.run_dir, "outputs/dinov3_grid_unfreeze2_raw_targets"
        )
        self.assertEqual(raw_config.model, normalized_config.model)
        self.assertEqual(raw_config.training, normalized_config.training)
        self.assertEqual(raw_config.augmentation, normalized_config.augmentation)
        self.assertEqual(raw_config.data.split_seed, normalized_config.data.split_seed)

    def test_four_block_config_only_changes_unfrozen_depth_and_output_identity(self):
        two_blocks = load_config(
            "experiments/dinov3_grid_unfreeze2/config.toml"
        )
        four_blocks = load_config(
            "experiments/dinov3_grid_unfreeze2/config_unfreeze4.toml"
        )

        self.assertTrue(four_blocks.data.normalize_targets)
        self.assertEqual(four_blocks.model.unfreeze_last_n_blocks, 4)
        self.assertEqual(two_blocks.training, four_blocks.training)
        self.assertEqual(two_blocks.augmentation, four_blocks.augmentation)
        self.assertEqual(two_blocks.runtime, four_blocks.runtime)
        self.assertEqual(two_blocks.data, four_blocks.data)
        self.assertEqual(
            four_blocks.output.run_dir,
            "outputs/dinov3_grid_unfreeze4",
        )
        self.assertEqual(
            {**two_blocks.model.__dict__, "unfreeze_last_n_blocks": 4},
            four_blocks.model.__dict__,
        )

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
