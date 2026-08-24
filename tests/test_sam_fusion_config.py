import tempfile
import unittest
from pathlib import Path

from experiments.dinov3_grid_lora_patch_attention.config import (
    load_config as load_lora_attention_config,
)
from experiments.dinov3_grid_lora_patch_attention_sam_fusion.config import load_config


class SamFusionConfigTests(unittest.TestCase):
    def test_preserves_the_lora_patch_attention_control(self):
        sam = load_config(
            "experiments/dinov3_grid_lora_patch_attention_sam_fusion/config.toml"
        )
        control = load_lora_attention_config(
            "experiments/dinov3_grid_lora_patch_attention/config.toml"
        )

        self.assertEqual(sam.data.__dict__, control.data.__dict__)
        self.assertEqual(sam.augmentation.__dict__, control.augmentation.__dict__)
        self.assertEqual(sam.runtime.__dict__, control.runtime.__dict__)
        for key in (
            "backbone",
            "processor",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "train_final_norm",
            "attention_hidden_dim",
            "attention_dropout",
            "attention_temperature",
            "head_hidden_dim",
            "dropout",
        ):
            self.assertEqual(getattr(sam.model, key), getattr(control.model, key))

        sam_training = sam.training.__dict__.copy()
        control_training = control.training.__dict__.copy()
        sam_batch = sam_training.pop("batch_size")
        sam_accumulation = sam_training.pop("gradient_accumulation_steps")
        warm_start_frozen_epochs = sam_training.pop("warm_start_frozen_epochs")
        base_auxiliary_loss_weight = sam_training.pop("base_auxiliary_loss_weight")
        delta_penalty_weight = sam_training.pop("delta_penalty_weight")
        control_batch = control_training.pop("batch_size")
        control_accumulation = control_training.pop("gradient_accumulation_steps")
        self.assertEqual(sam_training, control_training)
        self.assertEqual(
            sam_batch * sam_accumulation,
            control_batch * control_accumulation,
        )
        self.assertEqual(warm_start_frozen_epochs, 5)
        self.assertEqual(base_auxiliary_loss_weight, 0.25)
        self.assertEqual(delta_penalty_weight, 0.01)

    def test_three_representation_defaults_are_small_and_explicit(self):
        config = load_config(
            "experiments/dinov3_grid_lora_patch_attention_sam_fusion/config.toml"
        )

        self.assertEqual(config.segmentation.model_name, "facebook/sam3")
        self.assertEqual(config.segmentation.prompts, ["green leaf"])
        self.assertEqual(config.model.mask_input_size, 56)
        self.assertEqual(config.model.mask_embedding_dim, 128)
        self.assertEqual(config.model.fusion_hidden_dim, 256)
        self.assertEqual(config.training.batch_size, 4)
        self.assertEqual(config.training.gradient_accumulation_steps, 4)
        self.assertEqual(config.training.warm_start_frozen_epochs, 5)
        self.assertEqual(config.training.base_auxiliary_loss_weight, 0.25)
        self.assertEqual(config.training.delta_penalty_weight, 0.01)
        self.assertTrue(config.data.normalize_targets)
        self.assertIn("warmstart_aux", config.output.run_dir)

    def test_output_is_separate_from_the_control(self):
        sam = load_config(
            "experiments/dinov3_grid_lora_patch_attention_sam_fusion/config.toml"
        )
        control = load_lora_attention_config(
            "experiments/dinov3_grid_lora_patch_attention/config.toml"
        )

        self.assertNotEqual(sam.output.run_dir, control.output.run_dir)
        self.assertNotEqual(
            sam.output.attention_arrays_name,
            control.output.attention_arrays_name,
        )

    def test_empty_sam_prompts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(
                '[data]\ndataset_dir = "/dataset"\n'
                "[segmentation]\nprompts = []\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "prompts"):
                load_config(path)

    def test_invalid_mask_coverage_bounds_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(
                '[data]\ndataset_dir = "/dataset"\n'
                "[segmentation]\n"
                "minimum_foreground_fraction = 0.7\n"
                "maximum_foreground_fraction = 0.6\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "foreground fraction"):
                load_config(path)

    def test_negative_auxiliary_loss_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(
                '[data]\ndataset_dir = "/dataset"\n'
                "[training]\nbase_auxiliary_loss_weight = -0.1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "base_auxiliary_loss_weight"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
