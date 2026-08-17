"""Extract and cluster representations from pretrained or experiment models."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hdbscan
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from rapeseed_damage.artifacts import environment_info, write_json
from rapeseed_damage.checkpointing import load_checkpoint
from rapeseed_damage.reproducibility import resolve_device, seed_everything

from .config import AnalysisSettings, RepresentationSettings, load_analysis_config


class RepresentationDataset(Dataset):
    def __init__(self, table, experiment_config, processor, data_module, preprocessing):
        self.table = table.reset_index(drop=True)
        self.config = experiment_config
        self.processor = processor
        self.image_path = data_module.image_path
        self.preprocessing = preprocessing
        self.grid_crop = None
        if preprocessing == "grid":
            package = data_module.__package__
            self.grid_crop = importlib.import_module(f"{package}.preprocessing").load_grid_crop

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        filename = str(row[self.config.data.filename_column])
        path = self.image_path(self.config, filename)
        try:
            if self.preprocessing == "grid":
                image = self.grid_crop(path)
            else:
                with Image.open(path) as source:
                    image = source.convert("RGB")
        except Exception as error:
            raise RuntimeError(
                f"Preprocessing {self.preprocessing!r} failed for {filename!r} at {path}"
            ) from error
        pixels = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return {
            "pixel_values": pixels,
            "filename": filename,
            "image_path": str(path),
            "target": float(row[self.config.data.target_column]),
        }


def load_model_and_data(settings: RepresentationSettings, device, limit: int):
    config_module = importlib.import_module(f"{settings.experiment_package}.config")
    data_module = importlib.import_module(f"{settings.experiment_package}.data")
    model_module = importlib.import_module(f"{settings.experiment_package}.model")
    checkpoint_module = importlib.import_module(
        f"{settings.experiment_package}.checkpoint"
    )
    config = config_module.load_config(settings.experiment_config)
    table = data_module.load_scores(config)
    if limit > 0:
        table = table.head(limit)
    processor = AutoImageProcessor.from_pretrained(config.model.processor)
    dataset = RepresentationDataset(
        table, config, processor, data_module, settings.preprocessing
    )
    model = model_module.DinoV3Regressor(config).to(device).eval()
    if settings.checkpoint:
        state = load_checkpoint(settings.checkpoint, device)
        checkpoint_module.validate_for(state, config)
        model.load_state_dict(state["model_state_dict"])
    return model, dataset


def extract_embeddings(model, dataset, settings, analysis, device):
    loader = DataLoader(
        dataset,
        batch_size=analysis.batch_size,
        shuffle=False,
        num_workers=analysis.num_workers,
        persistent_workers=analysis.num_workers > 0,
    )
    embeddings, filenames, paths, targets = [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            features = model.extract_features(pixels)
            if settings.feature == "head":
                features = model.regression_head[:-1](features)
            features = functional.normalize(features, p=2, dim=1)
            embeddings.append(features.cpu().float().numpy())
            filenames.extend(map(str, batch["filename"]))
            paths.extend(map(str, batch["image_path"]))
            targets.extend(batch["target"].numpy().tolist())
    return (
        np.concatenate(embeddings),
        filenames,
        paths,
        np.asarray(targets, dtype=np.float32),
    )


def cluster_embeddings(embeddings, analysis):
    component_count = min(
        analysis.pca_components,
        len(embeddings) - 1,
        embeddings.shape[1],
    )
    if component_count < 2:
        raise ValueError("Clustering requires at least three images")
    reduced = PCA(n_components=component_count, random_state=analysis.seed).fit_transform(
        embeddings
    )
    labels = hdbscan.HDBSCAN(
        min_cluster_size=analysis.min_cluster_size,
        min_samples=analysis.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(reduced)
    coordinates = PCA(n_components=2, random_state=analysis.seed).fit_transform(embeddings)
    return labels, coordinates


def representation_metrics(embeddings, labels, targets):
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, -np.inf)
    nearest_indices = similarities.argmax(axis=1)
    nearest_similarity = similarities[np.arange(len(embeddings)), nearest_indices]
    clusters = sorted(set(labels) - {-1})
    metrics = {
        "images": len(embeddings),
        "embedding_dimension": embeddings.shape[1],
        "clusters": len(clusters),
        "noise_fraction": float(np.mean(labels == -1)),
        "nearest_similarity_median": float(np.median(nearest_similarity)),
        "nearest_target_mae": float(np.mean(np.abs(targets - targets[nearest_indices]))),
        "silhouette": None,
    }
    non_noise = labels != -1
    if len(set(labels[non_noise])) > 1 and non_noise.sum() > len(set(labels[non_noise])):
        metrics["silhouette"] = float(
            silhouette_score(embeddings[non_noise], labels[non_noise], metric="cosine")
        )
    return metrics


def save_plot(coordinates, colors, title, color_label, path, categorical=False):
    figure, axis = plt.subplots(figsize=(9, 7))
    plot = axis.scatter(
        coordinates[:, 0], coordinates[:, 1], c=colors, cmap="tab20" if categorical else "viridis"
    )
    axis.set(xlabel="PCA 1", ylabel="PCA 2", title=title)
    axis.grid(alpha=0.2)
    colorbar = figure.colorbar(plot, ax=axis)
    colorbar.set_label(color_label)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run(config_path: str | Path):
    analysis, representations = load_analysis_config(config_path)
    seed_everything(analysis.seed, deterministic=True)
    device = resolve_device(analysis.device)
    root = Path(analysis.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "environment.json", environment_info(device))
    summaries = []
    cluster_labels = {}
    reference_filenames = None

    for settings in representations:
        print(f"Extracting {settings.name} ...", flush=True)
        model, dataset = load_model_and_data(settings, device, analysis.limit)
        embeddings, filenames, paths, targets = extract_embeddings(
            model, dataset, settings, analysis, device
        )
        if reference_filenames is None:
            reference_filenames = filenames
        elif filenames != reference_filenames:
            raise ValueError(
                f"{settings.name} does not contain the same ordered image set as the first run"
            )
        labels, coordinates = cluster_embeddings(embeddings, analysis)
        cluster_labels[settings.name] = labels
        metrics = representation_metrics(embeddings, labels, targets)
        destination = root / settings.name
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "embeddings.npz",
            embeddings=embeddings,
            filenames=np.asarray(filenames),
            targets=targets,
        )
        pd.DataFrame(
            {
                "filename": filenames,
                "image_path": paths,
                "target": targets,
                "cluster_id": labels,
                "pca_1": coordinates[:, 0],
                "pca_2": coordinates[:, 1],
            }
        ).to_csv(destination / "assignments.csv", index=False)
        write_json(destination / "metrics.json", metrics)
        save_plot(
            coordinates,
            labels,
            f"{settings.name}: HDBSCAN clusters",
            "Cluster",
            destination / "clusters.png",
            categorical=True,
        )
        save_plot(
            coordinates,
            targets,
            f"{settings.name}: damage score",
            "Mean damage score",
            destination / "damage_scores.png",
        )
        summaries.append({"representation": settings.name, **metrics})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(summaries).to_csv(root / "comparison.csv", index=False)
    representation_names = list(cluster_labels)
    agreement = pd.DataFrame(
        [
            [
                adjusted_rand_score(cluster_labels[first], cluster_labels[second])
                for second in representation_names
            ]
            for first in representation_names
        ],
        index=representation_names,
        columns=representation_names,
    )
    agreement.to_csv(root / "cluster_agreement.csv")
    write_json(root / "comparison.json", summaries)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(run(arguments.config), indent=2))


if __name__ == "__main__":
    main()
