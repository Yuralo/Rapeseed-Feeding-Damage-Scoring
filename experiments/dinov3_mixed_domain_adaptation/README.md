# Raw tiled DINOv3 domain adaptation

This package adapts DINOv3 to the 7,569 canonical unlabeled CSFB photographs without applying the
supervised task's grid cropper. Every usable image remains untouched on disk. Training reads its
EXIF-oriented raw pixels and samples one overlapping high-resolution tile per image and epoch.

The tile scale is chosen equally from 3x3 and 4x4 grids. Within that scale, 70% of selections are
weighted toward probable vegetation and 30% are uniform, preserving soil and acquisition diversity.
Tiles overlapping the bright collector card are rejected when possible. These masks only control
sampling; they never paint over or modify pixels.

Two augmented views of the same tile are passed through a LoRA student and frozen original DINOv3
teacher. Cross-view cosine distillation supplies the adaptation signal, while a smaller same-view
teacher anchor limits destructive representation drift.

## 1. Install

```bash
python -m pip install -r experiments/dinov3_mixed_domain_adaptation/requirements.txt
python -m pip install -e .
```

## 2. Mandatory 100-image audit

Do this before any full-dataset pass:

```bash
python -m experiments.dinov3_mixed_domain_adaptation.audit_inputs \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml \
  --sample-size 100
```

This selects exactly 100 images across every cohort, source folder, and both `IMG_*` and timestamp
filename families. It spans each capture sequence instead of taking the first 100 files. For every
sample it writes a separate JPEG containing the untouched raw image, sampling masks, candidate tile
boxes, and four actual tiles under:

`outputs/dinov3_mixed_domain_adaptation/audit_100/`

`index.csv` marks each image as strictly decoded, recovered from a truncated JPEG, or failed. The
audit does not write or modify the prepared manifest and automatically removes its own stale
previews. Open all 100 previews before continuing. Recoverable truncated files must be judged from
their pixels here, not accepted from a console warning alone.

The older contact-sheet source audit remains available as an optional dataset overview:

```bash
python -m experiments.dinov3_mixed_domain_adaptation.inspect_sources \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml \
  --samples-per-source 8
```

## 3. Validate the complete raw dataset only after approving the audit

```bash
python -m experiments.dinov3_mixed_domain_adaptation.prepare_inputs \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml \
  --full
```

The command refuses to make a full-dataset pass unless `--full` is supplied deliberately.

This fully decodes each canonical source once and writes
`outputs/dinov3_mixed_domain_adaptation/prepared_manifest.csv`. It does not detect grids, crop,
rectify, resize, or create an image cache. The manifest stores only the 25 candidate boxes and their
small vegetation/label sampling scores, preventing repeated mask analysis during every epoch.
Unreadable files are logged and excluded. Preparation continues while exclusions remain below the
configured 5% safety limit.

## 4. Optional second inspection from the completed manifest

```bash
python -m experiments.dinov3_mixed_domain_adaptation.inspect_preprocessing \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml \
  --samples-per-cohort 3
```

Open the separate JPEG previews under
`outputs/dinov3_mixed_domain_adaptation/tile_inspection/`. Each preview contains the untouched raw
image, vegetation/collector-card diagnostics, sampled tile boxes, and the actual high-resolution
tiles. Check that:

- both 3x3 and 4x4 tiles appear;
- plant-biased samples contain useful plant detail;
- uniform samples retain some soil and acquisition diversity;
- selected tiles avoid collector labels;
- no image has been perspective-warped or grid-cropped.

Do not train until these previews look correct. Tile choices change deterministically with the epoch,
so the backbone sees different regions while resumed runs remain reproducible.

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

Batch size 8 with two accumulation steps gives an effective batch size of 16 and is a conservative
starting point for a 24 GB RTX 3090. Check the reported peak CUDA memory after epoch 1.

## 6. Export and run the controlled downstream comparison

```bash
python -m experiments.dinov3_mixed_domain_adaptation.export_backbone \
  --config experiments/dinov3_mixed_domain_adaptation/config.toml
```

The export merges LoRA into DINOv3 and writes a standard Hugging Face model and processor under
`outputs/dinov3_mixed_domain_adaptation/adapted_backbone/`. Point both `features.backbone` and
`features.processor` in a copy of the successful supervised tiled-MIL configuration to that folder.
Keep the labeled split, seed, supervised preprocessing, MIL tiles, head, and optimizer unchanged so
the adapted backbone is the only experimental difference.
