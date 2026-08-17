import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from .config import load_config
from .data import image_path, load_scores
from .preprocessing import load_grid_crop



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--count", type=int, default=6)
    arguments = parser.parse_args()

    config = load_config(arguments.config)

    output_dir = Path(config.output.run_dir) / "grid_crops"
    output_dir.mkdir(parents=True, exist_ok=True)

    table = load_scores(config).head(arguments.count)
    figure, axes = plt.subplots(
        len(table),
        2,
        figsize=(14, 6 * len(table)),
        squeeze=False,
    )

    for index, (_, row) in enumerate(table.iterrows()):
        filename = str(row[config.data.filename_column])
        path = image_path(config, filename)

        with Image.open(path) as source:
            original = source.convert("RGB")

        cropped = load_grid_crop(path)

        # Original image
        axes[index, 0].imshow(original)
        axes[index, 0].set_title(f"Original: {filename}")
        axes[index, 0].axis("off")

        # Cropped image
        axes[index, 1].imshow(cropped)
        axes[index, 1].set_title(f"Grid crop: {filename}")
        axes[index, 1].axis("off")

        output_path = (
            output_dir
            / f"{Path(filename).stem}_grid_crop.jpg"
        )
        cropped.save(output_path, quality=95)

        print(f"Saved {output_path}")

    plt.close(figure)

if __name__ == "__main__":
    main()