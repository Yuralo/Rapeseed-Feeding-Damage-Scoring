import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from .config import load_config
from .data import image_path, load_scores
from .preprocessing import load_grid_crop, log_grid_failure



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--count", type=int, default=6)
    arguments = parser.parse_args()

    config = load_config(arguments.config)

    output_dir = Path(config.output.run_dir) / "grid_crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_log = Path(config.output.run_dir) / config.output.grid_failure_log

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

        # Original image
        axes[index, 0].imshow(original)
        axes[index, 0].set_title(f"Original: {filename}")
        axes[index, 0].axis("off")

        try:
            cropped = load_grid_crop(path)
        except Exception as error:
            log_grid_failure(
                failure_log,
                error=error,
                image_path=path,
                filename=filename,
                dataset_index=index,
            )
            axes[index, 1].text(
                0.5,
                0.5,
                f"Grid detection failed\n{type(error).__name__}: {error}",
                ha="center",
                va="center",
                wrap=True,
            )
            axes[index, 1].axis("off")
            print(f"FAILED {filename}; details appended to {failure_log}")
            continue

        axes[index, 1].imshow(cropped)
        axes[index, 1].set_title(f"Grid crop: {filename}")
        axes[index, 1].axis("off")
        output_path = output_dir / f"{Path(filename).stem}_grid_crop.jpg"
        cropped.save(output_path, quality=95)
        print(f"Saved {output_path}")

    figure.tight_layout()
    comparison_path = output_dir / "grid_crop_comparison.png"
    figure.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved comparison plot to {comparison_path}")

if __name__ == "__main__":
    main()
