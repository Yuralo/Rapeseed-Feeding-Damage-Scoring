from pathlib import Path
import pandas as pd

SCORES_FILE = "RSFB-Phenotyping_training_set_scores.csv"


def load_data(dataset_path: str, scores_file: str = SCORES_FILE) -> pd.DataFrame:
    """Load and validate the image-score table."""
    data = pd.read_csv(Path(dataset_path) / scores_file)
    required_columns = {"Filename", "mean_score"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"The CSV is missing column(s): {', '.join(sorted(missing))}")
    return data


def image_path(dataset_path: Path, filename: str) -> Path:
    """Use the name from the CSV, allowing either names with or without .jpg."""
    path = dataset_path / filename
    return path if path.suffix else path.with_suffix(".jpg")
