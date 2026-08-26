"""Save one lightweight visual audit file per SAM-guided adaptive crop layout."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.dinov3_grid_lora_patch_attention_sam_fusion.preprocessing import (
    load_or_create_grid_crop,
)
from experiments.dinov3_grid_lora_patch_attention_sam_fusion.segmentation import load_cached_mask
from experiments.dinov3_grid_tiled_mil.data import image_path, load_scores
from rapeseed_damage.artifacts import write_json

from .config import load_config
from .crops import make_adaptive_crop_layout


def run(config, *, limit=24):
    table = load_scores(config).iloc[:limit]
    destination = Path(config.output.run_dir) / config.output.inspection_dir
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in table.iterrows():
        filename = str(row[config.data.filename_column])
        source = image_path(config, filename)
        image, _, _ = load_or_create_grid_crop(
            source,
            config.data.grid_cache_dir,
            size=config.data.grid_crop_size,
            inner_margin_fraction=config.data.grid_inner_margin_fraction,
        )
        mask, _, _ = load_cached_mask(source, config)
        layout = make_adaptive_crop_layout(np.asarray(mask) >= 128, config.adaptive_crops)
        canvas = image.convert("RGB").copy()
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay.putalpha(mask.point(lambda value: 70 if value >= 128 else 0))
        green = Image.new("RGBA", canvas.size, (0, 255, 0, 0))
        green.putalpha(overlay.getchannel("A"))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), green).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for index, box in enumerate(layout.boxes):
            draw.rectangle(tuple(map(int, box)), outline="cyan", width=5)
            draw.text((int(box[0]) + 8, int(box[1]) + 8), str(index), fill="cyan")
        output = destination / f"{Path(filename).stem}_adaptive_crops.jpg"
        canvas.save(output, quality=88, optimize=True)
        records.append(
            {
                "filename": filename,
                "instances": len(layout.boxes),
                "components_before_merge": layout.component_count_before_merge,
                "mask_coverage": layout.mask_coverage,
                "output": str(output),
            }
        )
        image.close()
        mask.close()
    report = {"images": len(records), "destination": str(destination), "records": records}
    write_json(destination / "manifest.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args(argv)
    print(json.dumps(run(load_config(args.config), limit=args.limit), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
