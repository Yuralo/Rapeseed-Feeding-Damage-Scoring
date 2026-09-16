from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path("outputs/random_gold_sample/images")
OUT = Path("tmp/damage_crops")
OUT.mkdir(parents=True, exist_ok=True)

IMAGES = [
    ("img_77b72ac3cd4ffb7f437c.jpg", "reference 3.5"),
    ("img_099d985bed2e812ed19e.jpg", "reference 10.0"),
    ("img_a5408dc361d9e1d21ec9.jpg", "target"),
    ("img_dd512f11e109aec10ced.jpg", "reference 22.25"),
    ("img_e5796f057776195b4605.jpg", "reference 14.5"),
]


def plant_centers(path: Path):
    full = Image.open(path).convert("RGB")
    scale = 0.25
    small = full.resize((int(full.width * scale), int(full.height * scale)))
    rgb = np.asarray(small)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Young rapeseed tissue is distinctly greener than red/blue soil pixels.
    mask = (g > 65) & (g > r * 1.08) & (g > b * 1.12)
    # Group the two cotyledons and nearby true-leaf pixels into one plant candidate.
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    grouped = np.asarray(mask_img.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.MinFilter(9))) > 0
    # Flood-fill connected components on the downsampled mask.
    seen = np.zeros(grouped.shape, dtype=bool)
    components = []
    for sy, sx in zip(*np.nonzero(grouped & ~seen)):
        if seen[sy, sx]:
            continue
        stack = [(int(sy), int(sx))]
        seen[sy, sx] = True
        points = []
        while stack:
            yy, xx = stack.pop()
            points.append((yy, xx))
            for ny, nx in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                if 0 <= ny < grouped.shape[0] and 0 <= nx < grouped.shape[1] and grouped[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx] = True
                    stack.append((ny,nx))
        components.append(points)
    candidates = []
    h, w = mask.shape
    for points in components:
        ys = [p[0] for p in points]
        xs = [p[1] for p in points]
        x, y = min(xs), min(ys)
        bw, bh = max(xs)-x+1, max(ys)-y+1
        area = len(points)
        if area < 20 or bw > 125 or bh > 125:
            continue
        cx, cy = sum(xs)/area, sum(ys)/area
        if x < 5 or y < 5 or x + bw > w - 5 or y + bh > h - 5:
            continue
        green_pixels = int(mask[y:y+bh, x:x+bw].sum())
        candidates.append((green_pixels, int(cx / scale), int(cy / scale)))
    return sorted(candidates, reverse=True)


def make_sheet(path: Path, label: str):
    image = Image.open(path).convert("RGB")
    centers = plant_centers(path)[:18]
    tile = 360
    header = 70
    cols = 6
    rows = (len(centers) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, header + rows * tile), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 18), f"{path.stem} — {label} — {len(centers)} largest detected plant clusters", fill="black")
    for idx, (_, cx, cy) in enumerate(centers):
        half = 150
        crop = image.crop((cx - half, cy - half, cx + half, cy + half)).resize((tile, tile))
        x = (idx % cols) * tile
        y = header + (idx // cols) * tile
        sheet.paste(crop, (x, y))
        ImageDraw.Draw(sheet).text((x + 8, y + 8), str(idx + 1), fill="red", stroke_width=2, stroke_fill="white")
    out = OUT / f"{path.stem}_crops.jpg"
    sheet.save(out, quality=94)
    print(out, len(centers))


for filename, label in IMAGES:
    make_sheet(ROOT / filename, label)
