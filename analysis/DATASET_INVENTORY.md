# Full CSFB dataset inventory

`analysis.dataset_inventory` recursively audits the complete image collection without modifying the
dataset. It ignores `__MACOSX`, assigns documented acquisition metadata, reads image headers, hashes
file contents to identify copied images, inventories score CSVs, and writes image/cohort/duplicate
reports. Optional OpenCV QR decoding adds provisional plot groups.

Run a small smoke test on the storage machine:

```bash
python -m analysis.dataset_inventory \
  --root /home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage \
  --output-dir outputs/dataset_inventory_smoke \
  --limit 20 \
  --decode-qr
```

Then run the complete audit. Do not use `--limit` or `--skip-hash` for the report that will be used
to design the adaptation experiment:

```bash
python -m analysis.dataset_inventory \
  --root /home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage \
  --output-dir outputs/dataset_inventory \
  --decode-qr
```

The command only reads the dataset. SHA-256 hashing requires reading the approximately 65 GB once,
and QR decoding opens every image, so the full run can take time. Progress is printed every 100
images. If OpenCV is unavailable, the remaining inventory still completes and records
`backend_unavailable` in the QR status column.

Outputs:

- `dataset_images.csv`: one row per discovered image path with cohort, dimensions, hashes,
  duplicate/canonical status, label availability, and optional QR plot grouping;
- `dataset_cohorts.csv`: counts and quality diagnostics by acquisition cohort;
- `dataset_duplicates.csv`: every exact duplicate group and its canonical path;
- `dataset_score_files.csv`: score-file row counts and column names;
- `dataset_summary.json`: compact totals, top-level folder counts, warnings, and the documented
  8,946-versus-9,456 discrepancy.

Return at least `dataset_images.csv`, `dataset_cohorts.csv`, `dataset_score_files.csv`, and
`dataset_summary.json`. Return `dataset_duplicates.csv` as well if it is not too large.

## Build supervised and adaptation manifests

After the complete inventory has been returned, join the four score tables to canonical images and
create plot-grouped manifests:

```bash
python -m analysis.build_supervised_manifests \
  --root /home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage \
  --inventory outputs/dataset_inventory/dataset_images.csv \
  --output-dir outputs/dataset_manifests \
  --gold-difference-threshold 5 \
  --seed 42
```

The builder uses score-table QR/plot metadata for grouping. Image-decoded QR is only a fallback.
This is not needed to identify filenames; it prevents the three views of one physical plot, or the
same plot at two timepoints, from crossing train and holdout partitions.

Supervision is kept in separate tiers:

- `gold`: the curated 470-image JLU/GAU calibration subset;
- `dual_weak`: other Gross-Gerau images with two subjective scores;
- `single_weak`: DSV and WG images with one visual score.

Validation and test contain only gold images. The remaining labels are written to `pretrain.csv`,
while gold training images are written to `finetune.csv`. `adaptation.csv` contains canonical images
that are safe for a later self-supervised experiment; both Gross-Gerau insect timepoints are
conservatively excluded by default because cross-timepoint plot linkage is incomplete.
