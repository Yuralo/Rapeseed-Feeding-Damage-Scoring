import tempfile
import unittest
from pathlib import Path

from experiments.dinov3_grid_patch_attention.config import load_config
from experiments.dinov3_grid_unfreeze2.config import load_config as load_two_block_config


class PatchAttentionConfigTests(unittest.TestCase):
    def test_config_preserves_winning_two_block_recipe(self):
        patch = load_config("experiments/dinov3_grid_patch_attention/config.toml")
        baseline = load_two_block_config("experiments/dinov3_grid_unfreeze2/config.toml")

        self.assertTrue(patch.data.normalize_targets)
        self.assertEqual(patch.model.unfreeze_last_n_blocks, 2)
        self.assertTrue(patch.model.unfreeze_final_norm)
        self.assertEqual(patch.data.__dict__, baseline.data.__dict__)
        self.assertEqual(patch.augmentation.__dict__, baseline.augmentation.__dict__)
        self.assertEqual(patch.training.__dict__, baseline.training.__dict__)
        self.assertEqual(patch.runtime.__dict__, baseline.runtime.__dict__)
        for key in (
            "backbone",
            "processor",
            "unfreeze_last_n_blocks",
            "unfreeze_final_norm",
            "head_hidden_dim",
            "dropout",
        ):
            self.assertEqual(getattr(patch.model, key), getattr(baseline.model, key))

    def test_attention_defaults_are_small_and_explicit(self):
        config = load_config("experiments/dinov3_grid_patch_attention/config.toml")

        self.assertEqual(config.model.attention_hidden_dim, 128)
        self.assertEqual(config.model.attention_temperature, 1.0)
        self.assertEqual(config.output.run_dir, "outputs/dinov3_grid_patch_attention")
        self.assertEqual(config.output.attention_inspection_images, 6)
        self.assertEqual(config.output.attention_top_fraction, 0.1)
        self.assertEqual(config.output.attention_ratio_min, 0.5)
        self.assertEqual(config.output.attention_ratio_max, 2.0)

    def test_clean_inset_run_matches_clean_two_block_control(self):
        patch = load_config(
            "experiments/dinov3_grid_patch_attention/config_clean_inset.toml"
        )
        baseline = load_two_block_config(
            "experiments/dinov3_grid_unfreeze2/config_clean_inset.toml"
        )

        self.assertEqual(patch.data.grid_inner_margin_fraction, 0.075)
        self.assertEqual(patch.data.__dict__, baseline.data.__dict__)
        self.assertEqual(patch.augmentation.__dict__, baseline.augmentation.__dict__)
        self.assertEqual(patch.training.__dict__, baseline.training.__dict__)
        self.assertEqual(patch.runtime.__dict__, baseline.runtime.__dict__)
        self.assertEqual(patch.model.unfreeze_last_n_blocks, 2)
        self.assertNotEqual(patch.output.run_dir, baseline.output.run_dir)

    def test_raw_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(
                '[data]\ndataset_dir = "/dataset"\nnormalize_targets = false\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "normalize_targets=true"):
                load_config(path)

    def test_invalid_attention_scale_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(
                '[data]\ndataset_dir = "/dataset"\n'
                "[output]\nattention_ratio_min = 1.0\nattention_ratio_max = 2.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "uniform value 1"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
