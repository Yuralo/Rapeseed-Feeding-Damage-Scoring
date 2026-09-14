# Source-routed DINOv3 domain adaptation

This controlled adaptation retry changes only the image routing relative to the previous raw-tiled
run. It starts from the original `facebook/dinov3-vitb16-pretrain-lvd1689m` backbone and retains the
same 3x3/4x4 tile sampling, paired augmentations, LoRA student, frozen teacher, and objective.

The source audit identified three folders where the universal grid detector repeatedly selected
only part of the quadrat. Those folders use the complete EXIF-oriented image:

- `2025_09_15_Re4StRes_T1_DSV`
- `2025_09_12_RSFB_01_NPZi`
- `2025_09_19_RSFB_02_NPZi`

Every other source uses the perspective-corrected 1400x1400 grid crop with a 7.5% inset. Routing is
by exact source folder, never by the `IMG_*` filename prefix. DSV T2 has `IMG_*` names but passed the
crop audit, so it remains on the cropped route.

Preparation caches one JPEG per successfully cropped source image. Raw-route images are referenced
directly and are never copied or modified. Grid failures are excluded and written with tracebacks to
`outputs/dinov3_routed_source_adaptation/input_exclusions.jsonl`; the run aborts if exclusions exceed
5% of the adaptation manifest.

## 1. Prepare the routed dataset

```bash
python -m pip install -r experiments/dinov3_routed_source_adaptation/requirements.txt
python -m pip install -e .

python -m experiments.dinov3_routed_source_adaptation.prepare_inputs \
  --config experiments/dinov3_routed_source_adaptation/config.toml \
  --full
```

Check `preparation_summary.json`. Its `source_routes` must show the three folders above only as
`raw_full_tiled`; every other folder must be `grid_inset075_tiled`.

## 2. Inspect the exact routed inputs before training

```bash
python -m experiments.dinov3_routed_source_adaptation.inspect_preprocessing \
  --config experiments/dinov3_routed_source_adaptation/config.toml \
  --samples-per-source 3
```

Open the separate previews under
`outputs/dinov3_routed_source_adaptation/routed_input_inspection/`. Each preview shows the original
source, the exact routed input, the vegetation/card diagnostics, and four tiles. Do not train if a
cropped route contains only half a quadrat.

## 3. Train or resume on the GPU machine

```bash
python -m experiments.dinov3_routed_source_adaptation.train \
  --config experiments/dinov3_routed_source_adaptation/config.toml \
  --from-scratch
```

```bash
python -m experiments.dinov3_routed_source_adaptation.train \
  --config experiments/dinov3_routed_source_adaptation/config.toml \
  --resume outputs/dinov3_routed_source_adaptation/last.pt
```

The experiment is configured for all 20 epochs so its downstream comparison is not decided by a
short self-supervised early-stopping window.

## 4. Export and test the adapted backbone

```bash
python -m experiments.dinov3_routed_source_adaptation.export_backbone \
  --config experiments/dinov3_routed_source_adaptation/config.toml
```

Then build the dedicated 3x3 and 4x4 labeled feature caches and train the unchanged multiscale MIL
head:

```bash
python -m experiments.dinov3_grid_tiled_mil.prepare_features \
  --config experiments/dinov3_grid_tiled_mil/config_adapted_routed_3x3.toml

python -m experiments.dinov3_grid_tiled_mil.prepare_features \
  --config experiments/dinov3_grid_tiled_mil/config_adapted_routed_4x4.toml

python -m experiments.dinov3_grid_multiscale_tiled_mil.train \
  --config experiments/dinov3_grid_multiscale_tiled_mil/config_adapted_routed.toml \
  --from-scratch
```

These configurations preserve the supervised split, seed, grid crop, feature representation, MIL
head, and optimizer. The exported routed-adaptation backbone is the only downstream model change.

