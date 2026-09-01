"""Visualize the exact raw-image tiles used by domain adaptation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rapeseed_damage.artifacts import write_json

from .config import Config, load_config
from .preprocessing import (
    choose_adaptation_tile,
    deserialize_tile_candidates,
    load_prepared_image,
    selection_masks,
)


def _read_manifest(config: Config) -> list[dict[str, str]]:
    path = Path(config.data.prepared_manifest)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared manifest is missing: {path}. Run the package prepare_inputs command first."
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "file_name",
        "cohort_id",
        "source_path",
        "input_mode",
        "tile_candidates",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    if not rows:
        raise ValueError(f"Prepared manifest is empty: {path}")
    return rows


def _sample(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    by_cohort: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cohort[row["cohort_id"]].append(row)
    selected = []
    for cohort, candidates in sorted(by_cohort.items()):
        rng = Random(f"{seed}:{cohort}")
        candidates = candidates.copy()
        rng.shuffle(candidates)
        selected.extend(candidates[: min(count, len(candidates))])
    return selected


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = image.copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    left = (size[0] - result.width) // 2
    top = (size[1] - result.height) // 2
    canvas.paste(result, (left, top))
    result.close()
    return canvas


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    resized = image.resize(size, Image.Resampling.NEAREST)
    image.close()
    return resized


def _overlay(image: Image.Image, label_mask: np.ndarray, vegetation_mask: np.ndarray, selections):
    result = image.convert("RGBA")
    vegetation_alpha = _resize_mask(vegetation_mask, image.size).point(lambda value: value * 45 // 255)
    green = Image.new("RGBA", image.size, (0, 255, 60, 0))
    green.putalpha(vegetation_alpha)
    result = Image.alpha_composite(result, green)
    label_alpha = _resize_mask(label_mask, image.size).point(lambda value: value * 130 // 255)
    red = Image.new("RGBA", image.size, (255, 0, 0, 0))
    red.putalpha(label_alpha)
    result = Image.alpha_composite(result, red)
    draw = ImageDraw.Draw(result)
    colors = ("#00bfff", "#ffd000", "#ff5ce1", "#00ffb3")
    width = max(5, round(min(image.size) / 400))
    for index, selection in enumerate(selections):
        draw.rectangle(selection.box, outline=colors[index % len(colors)], width=width)
    vegetation_alpha.close()
    label_alpha.close()
    green.close()
    red.close()
    return result.convert("RGB")


def _save_preview(
    record: dict[str, str],
    config: Config,
    destination: Path,
    index: int,
    *,
    image: Image.Image | None = None,
):
    image = load_prepared_image(record) if image is None else image
    label_mask, vegetation_mask = selection_masks(image, config)
    candidates = deserialize_tile_candidates(record["tile_candidates"])
    selections = []
    tiles = []
    for tile_index in range(config.tiles.preview_tiles_per_image):
        rng = Random(f"{config.training.seed}:{record['image_id']}:preview:{tile_index}")
        selection = choose_adaptation_tile(candidates, config, rng)
        selections.append(selection)
        tiles.append(image.crop(selection.box))
    overlay = _overlay(image, label_mask, vegetation_mask, selections)
    font = ImageFont.load_default()
    panel_size = (600, 500)
    tile_size = (300, 260)
    columns = max(2, min(4, len(tiles)))
    tile_rows = (len(tiles) + columns - 1) // columns
    canvas_width = max(2 * panel_size[0], columns * tile_size[0])
    canvas_height = 76 + panel_size[1] + tile_rows * (tile_size[1] + 48)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{record['file_name']} | raw tiled input | cohort={record['cohort_id']}"
    draw.text((16, 12), title, fill="black", font=font)
    draw.text(
        (16, 34),
        "Green = probable vegetation; red = collector label; boxes = sampled tiles",
        fill="black",
        font=font,
    )
    draw.text((16, 54), "Left: untouched raw image | Right: sampling diagnostics", fill="black")
    raw_panel = _thumbnail(image, panel_size)
    overlay_panel = _thumbnail(overlay, panel_size)
    canvas.paste(raw_panel, (0, 76))
    canvas.paste(overlay_panel, (600, 76))
    raw_panel.close()
    overlay_panel.close()
    y0 = 76 + panel_size[1]
    for tile_index, (tile, selection) in enumerate(zip(tiles, selections, strict=True)):
        column, row = tile_index % columns, tile_index // columns
        x, y = column * tile_size[0], y0 + row * (tile_size[1] + 48)
        tile_panel = _thumbnail(tile, tile_size)
        canvas.paste(tile_panel, (x, y))
        tile_panel.close()
        draw.text(
            (x + 6, y + tile_size[1] + 5),
            f"tile {tile_index + 1}: {selection.grid_size}x{selection.grid_size} "
            f"r{selection.row} c{selection.column} | {selection.sampling_strategy}",
            fill="black",
            font=font,
        )
        draw.text(
            (x + 6, y + tile_size[1] + 24),
            f"vegetation={selection.vegetation_fraction:.3f} | "
            f"label overlap={selection.label_overlap_fraction:.3f}",
            fill="black",
            font=font,
        )
    safe_stem = Path(record["file_name"]).stem.replace("/", "_")
    output = destination / f"{index:03d}_{record['cohort_id']}_{safe_stem}.jpg"
    canvas.save(output, format="JPEG", quality=88, optimize=True)
    for tile in tiles:
        tile.close()
    overlay.close()
    image.close()
    return output, selections, float(label_mask.mean()), float(vegetation_mask.mean())


def run(config: Config, *, samples_per_cohort: int | None = None) -> dict:
    rows = _read_manifest(config)
    count = samples_per_cohort or config.output.samples_per_cohort
    if count < 1:
        raise ValueError("samples_per_cohort must be positive")
    selected = _sample(rows, count, config.training.seed)
    destination = Path(config.output.run_dir) / config.output.inspection_dir
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for index, record in enumerate(selected, start=1):
        output, selections, label_fraction, vegetation_fraction = _save_preview(
            record, config, destination, index
        )
        records.append(
            {
                **record,
                "preview_path": str(output.resolve()),
                "probable_label_fraction": label_fraction,
                "probable_vegetation_fraction": vegetation_fraction,
                "selected_tiles": json.dumps(
                    [
                        {
                            "grid_size": selection.grid_size,
                            "row": selection.row,
                            "column": selection.column,
                            "box": selection.box,
                            "sampling_strategy": selection.sampling_strategy,
                            "vegetation_fraction": selection.vegetation_fraction,
                            "label_overlap_fraction": selection.label_overlap_fraction,
                        }
                        for selection in selections
                    ],
                    sort_keys=True,
                ),
            }
        )
        print(f"[{index:02d}/{len(selected):02d}] {output}", flush=True)
    index_path = destination / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    all_tiles = [json.loads(record["selected_tiles"]) for record in records]
    flattened = [tile for group in all_tiles for tile in group]
    summary = {
        "preview_files": len(records),
        "cohorts": dict(Counter(row["cohort_id"] for row in records)),
        "tile_grid_sizes": dict(Counter(str(tile["grid_size"]) for tile in flattened)),
        "sampling_strategies": dict(
            Counter(tile["sampling_strategy"] for tile in flattened)
        ),
        "directory": str(destination.resolve()),
        "index_csv": str(index_path.resolve()),
        "source_images_modified": False,
        "input_pipeline": "raw image -> label-safe 3x3/4x4 tile -> paired augmentations",
    }
    write_json(destination / "summary.json", summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-cohort", "--samples-per-mode", type=int)
    arguments = parser.parse_args(argv)
    report = run(
        load_config(arguments.config), samples_per_cohort=arguments.samples_per_cohort
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
