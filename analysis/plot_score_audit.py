"""Audit score coverage and render reproducible dataset/scorer figures.

Uses the canonical inventory, joined score manifest, and the weak/gold manifest
from the current experiment. No source photographs are required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_INVENTORY = Path("outputs/dataset_inventory/dataset_summary.json")
DEFAULT_SCORED = Path("outputs/dataset_manifests/scored_manifest.csv")
DEFAULT_WEAK = Path(
    "outputs/dinov3_weak_only_gold_validation_all_weak/manifests/weak_train.csv"
)
DEFAULT_OUTPUT = Path("outputs/dataset_score_audit")

INK = "#172635"
MUTED = "#516274"
GRID = "#D9E1E7"
GREY = "#9EADB9"
BLUE = "#3E76A8"
ORANGE = "#D48432"
TEAL = "#188D88"
BG = "#FFFFFF"


def _font(size: int, *, bold: bool = False):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _number(value: str) -> float | None:
    return None if value is None or not str(value).strip() else float(value)


def _mean(values):
    return statistics.mean(values) if values else None


def _median(values):
    return statistics.median(values) if values else None


def _correlation(a, b):
    if len(a) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    top = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return top / den if den else None


def build_summary(inventory: dict, scored: list[dict], weak: list[dict]) -> dict:
    """Calculate disjoint image/label counts and dual-scorer agreement statistics."""
    canonical = int(inventory["totals"]["canonical_image_paths"])
    tiers = Counter(row["supervision_tier"] for row in scored)
    if len({row["image_id"] for row in scored}) != len(scored):
        raise ValueError("Scored manifest contains duplicate image IDs")
    if sum(tiers.values()) != len(scored) or len(scored) > canonical:
        raise ValueError("Scored totals do not reconcile with the canonical inventory")
    dual = [row for row in scored if _number(row["score_jlu"]) is not None
            and _number(row["score_gau"]) is not None]
    if len(dual) != tiers["gold"] + tiers["dual_weak"]:
        raise ValueError("Dual-scorer availability and supervision tiers disagree")
    threshold = 5.0
    if any(abs(float(row["score_jlu"]) - float(row["score_gau"])) > threshold
           for row in dual if row["supervision_tier"] == "gold"):
        raise ValueError("Gold row exceeds the five-point disagreement threshold")
    if any(abs(float(row["score_jlu"]) - float(row["score_gau"])) <= threshold
           for row in dual if row["supervision_tier"] == "dual_weak"):
        raise ValueError("Dual-weak row falls within the gold threshold")
    cohort_tiers = defaultdict(Counter)
    for row in scored:
        cohort_tiers[row["cohort_id"]][row["supervision_tier"]] += 1
    cohorts = []
    for item in inventory["cohorts"]:
        if not item["canonical_images"]:
            continue  # The gold training folder is a duplicate copy, not another cohort.
        name = item["cohort_id"]
        counts = cohort_tiers[name]
        cohorts.append({
            "cohort_id": name,
            "canonical": int(item["canonical_images"]),
            "single_weak": counts["single_weak"],
            "dual_weak": counts["dual_weak"],
            "gold": counts["gold"],
            "unscored": int(item["canonical_images"]) - sum(counts.values()),
        })
    if sum(row["canonical"] for row in cohorts) != canonical:
        raise ValueError("Cohort image counts do not sum to the canonical inventory")
    if sum(row["unscored"] for row in cohorts) != canonical - len(scored):
        raise ValueError("Unscored cohort counts do not reconcile")
    weak_tiers = Counter(row["target_source"] for row in weak)
    if {row["image_id"] for row in weak} & {
        row["image_id"] for row in scored if row["supervision_tier"] == "gold"
    }:
        raise ValueError("Gold image appears in weak training manifest")
    scorer = {}
    for label, rows in (
        ("all_dual", dual),
        ("gold", [row for row in dual if row["supervision_tier"] == "gold"]),
        ("dual_weak", [row for row in dual if row["supervision_tier"] == "dual_weak"]),
    ):
        jlu = [float(row["score_jlu"]) for row in rows]
        gau = [float(row["score_gau"]) for row in rows]
        signed = [a - b for a, b in zip(jlu, gau, strict=True)]
        scorer[label] = {
            "images": len(rows),
            "jlu_mean": _mean(jlu),
            "gau_mean": _mean(gau),
            "jlu_minus_gau_mean": _mean(signed),
            "absolute_difference_mean": _mean([abs(value) for value in signed]),
            "absolute_difference_median": _median([abs(value) for value in signed]),
            "pearson_correlation": _correlation(jlu, gau),
            "jlu_higher": sum(value > 0 for value in signed),
            "gau_higher": sum(value < 0 for value in signed),
            "equal": sum(value == 0 for value in signed),
        }
    return {
        "discovered_image_paths": int(inventory["totals"]["discovered_image_paths"]),
        "duplicate_copies": int(inventory["totals"]["duplicate_copies"]),
        "canonical_images": canonical,
        "scored_images": len(scored),
        "unscored_images": canonical - len(scored),
        "supervision_tiers": dict(tiers),
        "dual_scored_images": len(dual),
        "weak_run_eligible_images": len(weak),
        "weak_run_target_sources": dict(weak_tiers),
        "cohorts": cohorts,
        "scorers": scorer,
        "gold_rule": "absolute JLU–GAU difference <= 5 score points, not 5% relative",
    }


def _header(draw: ImageDraw.ImageDraw, title: str, subtitle: str):
    draw.text((85, 52), title, font=_font(42, bold=True), fill=INK)
    draw.text((85, 110), subtitle, font=_font(23), fill=MUTED)


def _legend(draw, labels, y, start_x=85, gap=365):
    for index, (name, color) in enumerate(labels):
        x = start_x + index * gap
        draw.rounded_rectangle((x, y, x + 24, y + 24), radius=4, fill=color)
        draw.text((x + 37, y - 2), name, font=_font(22), fill=INK)


def plot_composition(summary: dict, destination: Path):
    image = Image.new("RGB", (1600, 850), BG)
    draw = ImageDraw.Draw(image)
    _header(draw, "Dataset and label coverage", "Unique images; duplicate folder copies counted once")
    tiers = summary["supervision_tiers"]
    cases = [
        ("All canonical images", [
            (summary["unscored_images"], GREY),
            (tiers["single_weak"], BLUE),
            (tiers["dual_weak"], ORANGE),
            (tiers["gold"], TEAL),
        ]),
        ("Scored images", [
            (tiers["single_weak"], BLUE), (tiers["dual_weak"], ORANGE),
            (tiers["gold"], TEAL),
        ]),
        ("Current weak run", [
            (summary["weak_run_target_sources"].get("single_scorer", 0), BLUE),
            (summary["weak_run_target_sources"].get("mean_of_two_scorers", 0), ORANGE),
        ]),
    ]
    for index, (label, parts) in enumerate(cases):
        y = 225 + index * 150
        total = sum(value for value, _ in parts)
        draw.text((85, y - 45), f"{label}  ·  {total:,}", font=_font(27, bold=True), fill=INK)
        left, width = 85, 1400
        for part_index, (value, color) in enumerate(parts):
            right = 85 + width * sum(item[0] for item in parts[:part_index + 1]) / total
            draw.rectangle((left, y, right, y + 65), fill=color)
            if right - left >= 105:
                foreground = INK if color == GREY else BG
                draw.text((left + 12, y + 16), f"{value:,}", font=_font(23, bold=True),
                          fill=foreground)
            left = right
    _legend(draw, [
        ("Unscored", GREY), ("One scorer", BLUE),
        ("Two, >5 apart", ORANGE), ("Gold, <=5 apart", TEAL),
    ], 710, gap=360)
    image.save(destination)


def _histogram(values, bins=20, maximum=50):
    result = [0] * bins
    for value in values:
        if 0 <= value <= maximum:
            result[min(int(value / maximum * bins), bins - 1)] += 1
    return [count / len(values) * 100 for count in result] if values else result


def plot_scores(scored: list[dict], weak: list[dict], destination: Path):
    gold = [row for row in scored if row["supervision_tier"] == "gold"]
    sources = [
        ("DSV, one scorer", "dsv_asendorf_t1_bbch11", BLUE),
        ("WG, one scorer", "wg_insects_t1_bbch10", "#658EC1"),
        ("GG, two scorers >5 apart", "gg_insects_t1_bbch10", ORANGE),
    ]
    groups = [(label, [float(row["target"]) for row in weak
                       if row["cohort_id"] == cohort], color)
              for label, cohort, color in sources]
    groups.append(("GG gold, <=5 apart", [float(row["target"]) for row in gold], TEAL))
    heights = [_histogram(values) for _, values, _ in groups]
    y_max = max(max(series) for series in heights) * 1.15
    image = Image.new("RGB", (1600, 1090), BG)
    draw = ImageDraw.Draw(image)
    _header(draw, "Score distributions by source", "Weak-run eligible labels compared with gold; each panel is percent of its group")
    for index, ((name, values, color), series) in enumerate(zip(groups, heights, strict=True)):
        column, row = index % 2, index // 2
        left, top = 110 + column * 760, 220 + row * 420
        right, bottom = left + 635, top + 270
        for tick in range(5):
            y = bottom - tick * (bottom - top) / 4
            draw.line((left, y, right, y), fill=GRID, width=2)
            draw.text((left - 62, y - 12), f"{y_max * tick / 4:.0f}%",
                      font=_font(19), fill=MUTED)
        for tick in (0, 10, 20, 30, 40, 50):
            x = left + (right - left) * tick / 50
            draw.text((x - 14, bottom + 9), str(tick), font=_font(19), fill=MUTED)
        for bucket, height in enumerate(series):
            x0 = left + bucket * (right - left) / 20 + 2
            x1 = left + (bucket + 1) * (right - left) / 20 - 2
            y = bottom - height / y_max * (bottom - top)
            draw.rectangle((x0, y, x1, bottom), fill=color)
        draw.line((left, top, left, bottom), fill=INK, width=2)
        draw.line((left, bottom, right, bottom), fill=INK, width=2)
        median = _median(values)
        median_label = f"{median:.1f}" if median is not None else "n/a"
        draw.text((left, top - 58),
                  f"{name}  ·  n={len(values):,}  ·  median={median_label}",
                  font=_font(25, bold=True), fill=INK)
        draw.text((left + 235, bottom + 39), "Target score (points)", font=_font(20), fill=INK)
    image.save(destination)


def _axes(draw, box, x_max, y_min, y_max, *, x_label, y_label, x_step=10, y_step=10):
    left, top, right, bottom = box
    for value in range(0, int(x_max) + 1, x_step):
        x = left + (right - left) * value / x_max
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((x - 16, bottom + 10), str(value), font=_font(18), fill=MUTED)
    for value in range(int(y_min), int(y_max) + 1, y_step):
        y = bottom - (bottom - top) * (value - y_min) / (y_max - y_min)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 52, y - 10), str(value), font=_font(18), fill=MUTED)
    draw.rectangle(box, outline=INK, width=2)
    draw.text((left + (right - left) / 2 - 95, bottom + 47), x_label,
              font=_font(22), fill=INK)
    draw.text((left, top - 48), y_label, font=_font(22), fill=INK)


def plot_scorers(scored: list[dict], summary: dict, destination: Path):
    dual = [row for row in scored if row["supervision_tier"] in {"gold", "dual_weak"}]
    image = Image.new("RGB", (1700, 980), BG)
    draw = ImageDraw.Draw(image)
    _header(draw, "Two scorers on the same images",
            f"Each dot is one of {len(dual):,} dual-scored images; "
            "gold means agreement within 5 score points")
    first = (130, 245, 760, 800)
    second = (980, 245, 1600, 800)
    _axes(draw, first, 70, 0, 70, x_label="JLU score", y_label="GAU score")
    _axes(draw, second, 45, -30, 70, x_label="Mean of two scores",
          y_label="JLU minus GAU", x_step=5, y_step=20)
    l, t, r, b = first
    for offset in (-5, 0, 5):
        points = []
        for x_value in range(71):
            y_value = x_value + offset
            if 0 <= y_value <= 70:
                points.append((l + (r - l) * x_value / 70,
                               b - (b - t) * y_value / 70))
        draw.line(points, fill=INK if offset == 0 else GREY, width=3 if offset == 0 else 2)
    l, t, r, b = second
    for difference in (-5, 0, 5):
        y = b - (b - t) * (difference + 30) / 100
        draw.line((l, y, r, y), fill=INK if difference == 0 else GREY,
                  width=3 if difference == 0 else 2)
    marks = Image.new("RGBA", image.size, (0, 0, 0, 0))
    dots = ImageDraw.Draw(marks)
    for row in sorted(dual, key=lambda item: item["supervision_tier"] == "gold"):
        jlu, gau = float(row["score_jlu"]), float(row["score_gau"])
        average, difference = (jlu + gau) / 2, jlu - gau
        color = (24, 141, 136, 130) if row["supervision_tier"] == "gold" else (212, 132, 50, 125)
        x = first[0] + (first[2] - first[0]) * jlu / 70
        y = first[3] - (first[3] - first[1]) * gau / 70
        dots.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        x = second[0] + (second[2] - second[0]) * average / 45
        y = second[3] - (second[3] - second[1]) * (difference + 30) / 100
        dots.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    image = Image.alpha_composite(image.convert("RGBA"), marks).convert("RGB")
    draw = ImageDraw.Draw(image)
    scorer = summary["scorers"]
    _legend(draw, [
        (f"Gold n={scorer['gold']['images']}, mean diff={scorer['gold']['jlu_minus_gau_mean']:+.1f}", TEAL),
        (f"Weak dual n={scorer['dual_weak']['images']}, mean diff={scorer['dual_weak']['jlu_minus_gau_mean']:+.1f}", ORANGE),
    ], 900, gap=760)
    image.save(destination)


def run(inventory_path: Path, scored_path: Path, weak_path: Path, output_dir: Path) -> dict:
    with inventory_path.open(encoding="utf-8") as handle:
        inventory = json.load(handle)
    scored, weak = _csv_rows(scored_path), _csv_rows(weak_path)
    summary = build_summary(inventory, scored, weak)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "cohort_breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "cohort_id", "canonical", "single_weak", "dual_weak", "gold", "unscored"
        ))
        writer.writeheader()
        writer.writerows(summary["cohorts"])
    plot_composition(summary, output_dir / "dataset_composition.png")
    plot_scores(scored, weak, output_dir / "score_distributions.png")
    plot_scorers(scored, summary, output_dir / "scorer_comparison.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--weak", type=Path, default=DEFAULT_WEAK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.inventory, args.scored, args.weak, args.output_dir)
    print(json.dumps({
        "canonical_images": result["canonical_images"],
        "scored_images": result["scored_images"],
        "supervision_tiers": result["supervision_tiers"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
