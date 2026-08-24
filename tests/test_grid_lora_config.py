import tempfile
import unittest
from pathlib import Path

from experiments.dinov3_grid_lora.config import load_config
from experiments.dinov3_grid_unfreeze2.config import load_config as load_two_block_config


class GridLoRAConfigTests(unittest.TestCase):
    def test_lora_run_uses_the_clean_two_block_data_control(self):
        lora = load_config("experiments/dinov3_grid_lora/config.toml")
        baseline = load_two_block_config(
            "experiments/dinov3_grid_unfreeze2/config_clean_inset.toml"
        )

        self.assertEqual(lora.data.__dict__, baseline.data.__dict__)
        self.assertEqual(lora.augmentation.__dict__, baseline.augmentation.__dict__)
        self.assertEqual(lora.runtime.__dict__, baseline.runtime.__dict__)
        self.assertEqual(lora.training.batch_size, baseline.training.batch_size)
        self.assertEqual(
            lora.training.gradient_accumulation_steps,
            baseline.training.gradient_accumulation_steps,
        )
        self.assertEqual(lora.training.seed, baseline.training.seed)

    def test_adapter_defaults_are_conservative_and_distributed(self):
        config = load_config("experiments/dinov3_grid_lora/config.toml")

        self.assertEqual(config.model.lora_rank, 8)
        self.assertEqual(config.model.lora_alpha, 16)
        self.assertEqual(config.model.lora_target_modules, ["q_proj", "v_proj"])
        self.assertTrue(config.model.train_final_norm)
        self.assertEqual(config.training.adapter_learning_rate, 1e-4)
        self.assertEqual(
            config.output.run_dir,
            "outputs/dinov3_grid_lora_clean_inset075",
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

    def test_empty_adapter_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(
                '[data]\ndataset_dir = "/dataset"\n'
                "[model]\nlora_target_modules = []\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "lora_target_modules"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
