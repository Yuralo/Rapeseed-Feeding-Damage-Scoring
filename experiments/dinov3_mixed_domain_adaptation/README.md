# Mixed-source DINOv3 domain adaptation

This experiment adapts DINOv3 to the unlabeled CSFB photographs without mixing incompatible
preprocessing rules:

- `YYYYMMDD_HHMMSS.jpg` uses the established grid detector, perspective crop, and 7.5% inset.
- `IMG_*.JPG` is already framed well and is used directly, with no homography.
- unknown filename styles fail explicitly; there is no silent preprocessing fallback.

Training selects a high-resolution local crop from each routed input. It rejects crops overlapping
the bright collector card when possible, but never paints over or modifies the underlying image.
Two augmented views of the same local content are passed through a LoRA student and frozen original
DINOv3 teacher. Cross-view cosine distillation supplies the adaptation signal; a smaller same-view
teacher anchor limits destructive representation drift.

## 1. Install this experiment

```bash
python -m pip install -r experiments/dinov3_mixed_domain_adaptation/requirements.txt
python -m pip install -e .
```

## 2. Inspect every raw source before deciding the routing

Do this before creating a single grid crop:

```bash
python -m experiments.dinov3_mixed_domain_adaptation.inspect_sources \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml \
  --samples-per-source 8
```

This writes one manageable contact sheet per cohort/source folder under
`outputs/dinov3_mixed_domain_adaptation/source_inspection/`. The samples are spread evenly across
the filename sequence rather than taken only from the beginning of a folder. These sheets contain
raw, EXIF-oriented images only: no crop, inset, square resize, or perspective warp is applied.
The display thumbnails are resized with their original aspect ratio preserved.

Review every sheet and decide which source needs:

- the established perspective-corrected grid crop;
- direct raw input;
- a simple fixed border crop; or
- a separate source-specific rule.

The route printed on a sheet is only the current filename-based proposal. It is deliberately marked
`visual_review_required`; change the configuration/code after reviewing the sheets if the proposal
is wrong. `index.csv` records every sampled filename and `sources.csv` provides one row per source.

## 3. Prepare the reviewed routes

```bash
python -m experiments.dinov3_mixed_domain_adaptation.prepare_inputs \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml
```

The source audit found 7,569 canonical images: approximately 5,769 `grid_crop` timestamp images and
1,800 raw `IMG_*` images. Preparation writes a resumable timestamp crop cache and
`outputs/dinov3_mixed_domain_adaptation/prepared_manifest.csv`. Individual unreadable files and
failed grid detections are logged and excluded; the command succeeds while exclusions remain below
the configured 5% safety limit. This avoids spending effort on a tiny unusable fraction while still
stopping if preprocessing is broken systemically. The raw images are referenced in place and are
not duplicated.

## 4. Inspect the routed preprocessing and local crops

```bash
python -m experiments.dinov3_mixed_domain_adaptation.inspect_preprocessing \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml \
  --samples-per-mode 20
```

Open the separate JPEG files under
`outputs/dinov3_mixed_domain_adaptation/preprocessing_inspection/`. Each preview shows the source,
the routed input, probable collector-label regions in red, and four example local training crops.
Check that:

- timestamp inputs are correctly rectified quadrats;
- `IMG_*` inputs remain unwarped;
- local boxes cover plants and damage at useful resolution;
- selected crops avoid the QR/collector card.

Do not start training until these previews look correct. Adjust only `[crops]` if the local crop
scale or card-overlap threshold needs tuning, then regenerate the previews.

## 5. Train on the GPU machine

From scratch:

```bash
python -m experiments.dinov3_mixed_domain_adaptation.train \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml \
  --from-scratch
```

Resume after interruption:

```bash
python -m experiments.dinov3_mixed_domain_adaptation.train \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml \
  --resume outputs/dinov3_mixed_domain_adaptation/last.pt
```

The configured batch size 8 with two accumulation steps has an effective batch size of 16 and is a
conservative starting point for a 24 GB RTX 3090. Check the reported peak CUDA memory after epoch 1
before increasing it.

## 6. Export a normal backbone

```bash
python -m experiments.dinov3_mixed_domain_adaptation.export_backbone \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml
```

This merges LoRA into DINOv3 and writes a standard Hugging Face model plus processor to
`outputs/dinov3_mixed_domain_adaptation/adapted_backbone/`. For the controlled downstream test, set
both `features.backbone` and `features.processor` in a copy of the successful 3x3 + 4x4 MIL config to
that directory. Keep the gold split, grid preprocessing, feature representation, and MIL head
unchanged so the backbone adaptation is the only experimental difference.
