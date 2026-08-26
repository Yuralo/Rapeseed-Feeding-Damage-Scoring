"""PIL-only paired-view augmentations for domain adaptation."""

from __future__ import annotations

from random import Random

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .config import Config


def augment_view(image: Image.Image, config: Config, rng: Random) -> Image.Image:
    settings = config.augmentation
    result = image.convert("RGB")
    if rng.random() < settings.horizontal_flip_probability:
        result = ImageOps.mirror(result)
    if rng.random() < settings.vertical_flip_probability:
        result = ImageOps.flip(result)
    strength = settings.color_jitter_strength
    if strength:
        result = ImageEnhance.Brightness(result).enhance(rng.uniform(1 - strength, 1 + strength))
        result = ImageEnhance.Contrast(result).enhance(rng.uniform(1 - strength, 1 + strength))
        result = ImageEnhance.Color(result).enhance(rng.uniform(1 - strength, 1 + strength))
    if rng.random() < settings.grayscale_probability:
        result = ImageOps.grayscale(result).convert("RGB")
    if settings.blur_max_radius and rng.random() < settings.blur_probability:
        result = result.filter(
            ImageFilter.GaussianBlur(radius=rng.uniform(0.1, settings.blur_max_radius))
        )
    return result


def paired_views(image: Image.Image, config: Config, rng: Random):
    """Create two augmentations of the same local content crop."""
    return augment_view(image, config, rng), augment_view(image, config, rng)
