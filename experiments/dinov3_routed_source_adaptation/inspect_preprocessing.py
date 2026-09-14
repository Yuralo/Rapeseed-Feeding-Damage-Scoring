"""Write separate previews of the exact routed inputs and sampled tiles."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

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
        raise FileNotFoundError(f"Prepared manifest is missing: {path}. Run prepare_inputs first.")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "file_name",
        "cohort_id",
        "source_folder",
        "source_path",
        "prepared_path",
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
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_folder"]].append(row)
    selected = []
    for source, candidates in sorted(by_source.items()):
        rng = Random(f"{seed}:{source}:routed-inspection")
        candidates = candidates.copy()
        rng.shuffle(candidates)
        selected.extend(candidates[: min(count, len(candidates))])
    return selected


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        image.load()
        return image.copy()


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    result = image.copy()
    result.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(result, ((size[0] - result.width) // 2, (size[1] - result.height) // 2))
    result.close()
    return canvas


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    resized = image.resize(size, Image.Resampling.NEAREST)
    image.close()
    return resized


def _overlay(image, label_mask, vegetation_mask, selections):
    result = image.convert("RGBA")
    vegetation_alpha = _resize_mask(vegetation_mask, image.size).point(
        lambda value: value * 45 // 255
    )
    vegetation = Image.new("RGBA", image.size, (0, 255, 60, 0))
    vegetation.putalpha(vegetation_alpha)
    result = Image.alpha_composite(result, vegetation)
    label_alpha = _resize_mask(label_mask, image.size).point(lambda value: value * 130 // 255)
    label = Image.new("RGBA", image.size, (255, 0, 0, 0))
    label.putalpha(label_alpha)
    result = Image.alpha_composite(result, label)
    draw = ImageDraw.Draw(result)
    colors = ("#00bfff", "#ffd000", "#ff5ce1", "#00ffb3")
    width = max(5, round(min(image.size) / 400))
    for index, selection in enumerate(selections):
        draw.rectangle(selection.box, outline=colors[index % len(colors)], width=width)
    for opened in (vegetation_alpha, vegetation, label_alpha, label):
        opened.close()
    return result.convert("RGB")


def _save_preview(record, config: Config, destination: Path, index: int):
    raw = _load_rgb(record["source_path"])
    routed = load_prepared_image(record)
    label_mask, vegetation_mask = selection_masks(routed, config)
    candidates = deserialize_tile_candidates(record["tile_candidates"])
    selections, tiles = [], []
    for tile_index in range(config.tiles.preview_tiles_per_image):
        rng = Random(f"{config.training.seed}:{record['image_id']}:preview:{tile_index}")
        selection = choose_adaptation_tile(candidates, config, rng)
        selections.append(selection)
        tiles.append(routed.crop(selection.box))
    overlay = _overlay(routed, label_mask, vegetation_mask, selections)

    panel_size = (480, 420)
    tile_size = (300, 250)
    columns = max(2, min(4, len(tiles)))
    tile_rows = (len(tiles) + columns - 1) // columns
    canvas_width = max(3 * panel_size[0], columns * tile_size[0])
    canvas_height = 96 + panel_size[1] + tile_rows * (tile_size[1] + 46)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (16, 12),
        f"{record['file_name']} | {record['input_mode']} | source={record['source_folder']}",
        fill="black",
        font=font,
    )
    draw.text(
        (16, 34),
        "Left: original source | middle: routed training input | right: masks and sampled boxes",
        fill="black",
        font=font,
    )
    draw.text(
        (16, 56),
        "Green = probable vegetation; red = collector card. No mask changes training pixels.",
        fill="black",
        font=font,
    )
    draw.text(
        (16, 76),
        f"source size={raw.width}x{raw.height} | routed size={routed.width}x{routed.height}",
        fill="black",
        font=font,
    )
    for column, image in enumerate((raw, routed, overlay)):
        panel = _thumbnail(image, panel_size)
        canvas.paste(panel, (column * panel_size[0], 96))
        panel.close()

    y0 = 96 + panel_size[1]
    for tile_index, (tile, selection) in enumerate(zip(tiles, selections, strict=True)):
        column, row = tile_index % columns, tile_index // columns
        x, y = column * tile_size[0], y0 + row * (tile_size[1] + 46)
        tile_panel = _thumbnail(tile, tile_size)
        canvas.paste(tile_panel, (x, y))
        tile_panel.close()
        draw.text(
            (x + 6, y + tile_size[1] + 5),
            f"{selection.grid_size}x{selection.grid_size} r{selection.row} c{selection.column} "
            f"| {selection.sampling_strategy}",
            fill="black",
            font=font,
        )
        draw.text(
            (x + 6, y + tile_size[1] + 23),
            f"vegetation={selection.vegetation_fraction:.3f} | "
            f"label={selection.label_overlap_fraction:.3f}",
            fill="black",
            font=font,
        )

    safe_source = record["source_folder"].replace("/", "_").replace("<", "").replace(">", "")
    safe_stem = Path(record["file_name"]).stem.replace("/", "_")
    output = destination / f"{index:03d}_{safe_source}_{safe_stem}.jpg"
    canvas.save(output, format="JPEG", quality=88, optimize=True)
    for opened in (raw, routed, overlay, *tiles, canvas):
        opened.close()
    return output, selections, float(label_mask.mean()), float(vegetation_mask.mean())


def run(config: Config, *, samples_per_source: int | None = None) -> dict:
    rows = _read_manifest(config)
    count = samples_per_source or config.output.samples_per_source
    if count < 1:
        raise ValueError("samples_per_source must be positive")
    selected = _sample(rows, count, config.training.seed)
    destination = Path(config.output.run_dir) / config.output.inspection_dir
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("*.jpg"):
        stale.unlink()
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
    flattened = [tile for record in records for tile in json.loads(record["selected_tiles"])]
    summary = {
        "preview_files": len(records),
        "input_modes": dict(Counter(row["input_mode"] for row in records)),
        "source_routes": dict(
            Counter(f"{row['source_folder']}|{row['input_mode']}" for row in records)
        ),
        "tile_grid_sizes": dict(Counter(str(tile["grid_size"]) for tile in flattened)),
        "sampling_strategies": dict(Counter(tile["sampling_strategy"] for tile in flattened)),
        "directory": str(destination.resolve()),
        "index_csv": str(index_path.resolve()),
        "source_images_modified": False,
        "input_pipeline": (
            "three audited sources -> raw full image; all others -> grid crop + 7.5% inset; "
            "then label-safe 3x3/4x4 tile -> paired augmentations"
        ),
    }
    write_json(destination / "summary.json", summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-source", type=int)
    arguments = parser.parse_args(argv)
    report = run(load_config(arguments.config), samples_per_source=arguments.samples_per_source)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
