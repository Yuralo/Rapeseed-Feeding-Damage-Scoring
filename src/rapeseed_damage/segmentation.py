"""Optional SAM3 plant segmentation extracted from the prototype notebook."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


class SamPlantSegmenter:
    def __init__(self, model_name: str = "facebook/sam3", device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.processor = Sam3Processor.from_pretrained(model_name)
        self.model = Sam3Model.from_pretrained(model_name).to(self.device).eval()

    def __call__(
        self,
        rgb_image,
        prompts: Sequence[str] = ("green leaf",),
        score_threshold: float = 0.25,
        mask_threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(rgb_image)
        if array.dtype != np.uint8:
            if array.size and array.max() <= 1:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        combined_mask = np.zeros((image.height, image.width), dtype=bool)

        for prompt in prompts:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
                self.device
            )
            with torch.inference_mode():
                outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=score_threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs["original_sizes"].tolist(),
            )[0]
            masks = result["masks"]
            if len(masks) > 0:
                combined_mask |= masks.any(dim=0).cpu().numpy().astype(bool)

        plant_only = np.full_like(array, 255)
        plant_only[combined_mask] = array[combined_mask]
        return combined_mask, plant_only

