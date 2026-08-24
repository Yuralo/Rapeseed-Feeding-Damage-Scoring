import tempfile
import unittest
from pathlib import Path

from experiments.dinov3_grid_lora.config import load_config as load_lora_config
from experiments.dinov3_grid_lora_patch_attention.config import load_config
from experiments.dinov3_grid_patch_attention.config import (
    load_config as load_patch_attention_config,
)


class GridLoRAPatchAttentionConfigTests(unittest.TestCase):
    def test_combined_run_preserves_the_lora_control_recipe(self):
        combined = load_config(
            "experiments/dinov3_grid_lora_patch_attention/config.toml"
        )
        lora = load_lora_config("experiments/dinov3_grid_lora/config.toml")

        self.assertEqual(combined.data.__dict__, lora.data.__dict__)
        self.assertEqual(combined.augmentation.__dict__, lora.augmentation.__dict__)
        self.assertEqual(combined.training.__dict__, lora.training.__dict__)
        self.assertEqual(combined.runtime.__dict__, lora.runtime.__dict__)
        for key in (
            "backbone",
            "processor",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "train_final_norm",
            "head_hidden_dim",
            "dropout",
        ):
            self.assertEqual(getattr(combined.model, key), getattr(lora.model, key))

    def test_gated_attention_settings_match_the_existing_attention_control(self):
        combined = load_config(
            "experiments/dinov3_grid_lora_patch_attention/config.toml"
        )
        attention = load_patch_attention_config(
            "experiments/dinov3_grid_patch_attention/config_clean_inset.toml"
        )

        for key in (
            "attention_hidden_dim",
            "attention_dropout",
            "attention_temperature",
        ):
            self.assertEqual(getattr(combined.model, key), getattr(attention.model, key))
        for key in (
            "attention_inspection_images",
            "attention_top_fraction",
            "attention_ratio_min",
            "attention_ratio_max",
            "attention_arrays_name",
        ):
            self.assertEqual(getattr(combined.output, key), getattr(attention.output, key))

    def test_output_and_required_model_settings_are_explicit(self):
        config = load_config(
            "experiments/dinov3_grid_lora_patch_attention/config.toml"
        )

        self.assertTrue(config.data.normalize_targets)
        self.assertEqual(config.model.lora_target_modules, ["q_proj", "v_proj"])
        self.assertEqual(config.model.attention_hidden_dim, 128)
        self.assertEqual(
            config.output.run_dir,
            "outputs/dinov3_grid_lora_patch_attention_clean_inset075",
        )

    def test_raw_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(
                '[data]\ndataset_dir = "/dataset"\nnormalize_targets = false\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "normalized targets"):
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
