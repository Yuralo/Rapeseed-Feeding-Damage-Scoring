"""Save one inspectable preview file per mixed-source adaptation image."""

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
from .preprocessing import load_prepared_image, probable_label_mask, select_local_crop


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
        "processed_path",
        "preprocessing_mode",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    if not rows:
        raise ValueError(f"Prepared manifest is empty: {path}")
    return rows


def _sample(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_mode[row["preprocessing_mode"]].append(row)
    selected = []
    for mode, candidates in sorted(by_mode.items()):
        rng = Random(f"{seed}:{mode}")
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
    return canvas


def _overlay(image: Image.Image, mask: np.ndarray, selections) -> Image.Image:
    result = image.convert("RGBA")
    mask_image = Image.fromarray((mask * 110).astype(np.uint8), mode="L")
    red = Image.new("RGBA", image.size, (255, 0, 0, 0))
    red.putalpha(mask_image)
    result = Image.alpha_composite(result, red)
    draw = ImageDraw.Draw(result)
    colors = ("#00ff66", "#00bfff", "#ffd000", "#ff5ce1")
    for index, selection in enumerate(selections):
        draw.rectangle(selection.box, outline=colors[index % len(colors)], width=8)
    return result.convert("RGB")


def _save_preview(record: dict[str, str], config: Config, destination: Path, index: int):
    with Image.open(record["source_path"]) as source_handle:
        source = source_handle.convert("RGB").copy()
    processed = load_prepared_image(record)
    label_mask = probable_label_mask(processed)
    selections = []
    crops = []
    for crop_index in range(config.crops.preview_crops_per_image):
        rng = Random(f"{config.training.seed}:{record['image_id']}:{crop_index}")
        selection = select_local_crop(processed, config, rng, label_mask=label_mask)
        selections.append(selection)
        crops.append(processed.crop(selection.box))
    font = ImageFont.load_default()
    panel_size = (600, 500)
    crop_size = (300, 260)
    columns = max(2, min(4, len(crops)))
    crop_rows = (len(crops) + columns - 1) // columns
    canvas_width = max(2 * panel_size[0], columns * crop_size[0])
    canvas_height = 72 + panel_size[1] + crop_rows * (crop_size[1] + 30)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{record['file_name']} | mode={record['preprocessing_mode']} | "
        f"cohort={record['cohort_id']}"
    )
    draw.text((16, 12), title, fill="black", font=font)
    draw.text(
        (16, 34),
        "Red = probable collector label; colored boxes = candidate training crops",
        fill="black",
        font=font,
    )
    canvas.paste(_thumbnail(source, panel_size), (0, 72))
    canvas.paste(_thumbnail(_overlay(processed, label_mask, selections), panel_size), (600, 72))
    draw.text((16, 54), "Left: source | Right: routed input", fill="black", font=font)
    y0 = 72 + panel_size[1]
    for crop_index, (crop, selection) in enumerate(zip(crops, selections, strict=True)):
        column, row = crop_index % columns, crop_index // columns
        x, y = column * crop_size[0], y0 + row * (crop_size[1] + 30)
        canvas.paste(_thumbnail(crop, crop_size), (x, y))
        draw.text(
            (x + 6, y + crop_size[1] + 5),
            f"crop {crop_index + 1} | label overlap={selection.label_overlap_fraction:.3f}",
            fill="black",
            font=font,
        )
    safe_stem = Path(record["file_name"]).stem.replace("/", "_")
    output = destination / f"{index:03d}_{record['preprocessing_mode']}_{safe_stem}.jpg"
    canvas.save(output, format="JPEG", quality=88, optimize=True)
    processed.close()
    return output, selections, float(label_mask.mean())


def run(config: Config, *, samples_per_mode: int | None = None) -> dict:
    rows = _read_manifest(config)
    count = samples_per_mode or config.output.samples_per_mode
    if count < 1:
        raise ValueError("samples_per_mode must be positive")
    selected = _sample(rows, count, config.training.seed)
    destination = Path(config.output.run_dir) / config.output.inspection_dir
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for index, record in enumerate(selected, start=1):
        output, selections, label_fraction = _save_preview(record, config, destination, index)
        records.append(
            {
                **record,
                "preview_path": str(output.resolve()),
                "probable_label_fraction": label_fraction,
                "maximum_selected_crop_label_overlap": max(
                    selection.label_overlap_fraction for selection in selections
                ),
            }
        )
        print(f"[{index:02d}/{len(selected):02d}] {output}", flush=True)
    index_path = destination / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "preview_files": len(records),
        "counts": dict(Counter(row["preprocessing_mode"] for row in records)),
        "directory": str(destination.resolve()),
        "index_csv": str(index_path.resolve()),
        "one_preview_per_file": True,
        "training_image_mutation": False,
        "label_handling": "reject local crops overlapping probable collector-card regions",
    }
    write_json(destination / "summary.json", summary)
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-mode", type=int)
    arguments = parser.parse_args(argv)
    report = run(load_config(arguments.config), samples_per_mode=arguments.samples_per_mode)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
