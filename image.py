from __future__ import annotations

import itertools
import math
from pathlib import Path

import cv2
import numpy as np


IMAGE_PATH = Path(
    "/home/nfs/data/nvme_datasets/Pictures_CFSB_leaf_damage/"
    "RSFB-Phenotyping_training_set/RSFB-Phenotyping_training_set/"
    "20251021_120905.jpg"
)

OUTPUT_DIR = Path("/home/user/tabbakhab1/dev/project/quadrats_output")

# Detection is performed on a resized copy for speed.
MAX_DETECTION_DIMENSION = 1600

# Final width and height of each extracted square.
CELL_SIZE = 900

# Removes the steel bars from the borders of each extracted cell.
INNER_MARGIN_FRACTION = 0.05

# Steel-color thresholds.
# Steel is generally less saturated than brown soil.
SATURATION_MAX = 60
VALUE_MIN = 70

# Maximum allowed deviation from horizontal or vertical.
ANGLE_TOLERANCE = 14.0


def resize_for_detection(
    image: np.ndarray,
    maximum_dimension: int,
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]

    scale = min(
        1.0,
        maximum_dimension / max(height, width),
    )

    if scale == 1.0:
        return image.copy(), scale

    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    return resized, scale


def detect_possible_steel(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finds low-saturation linear objects.

    Returns:
        steel_color_mask:
            Initial mask based on color.

        steel_line_mask:
            Mask containing long horizontal and vertical structures.
    """
    height, width = image.shape[:2]
    minimum_dimension = min(height, width)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    steel_color_mask = (
        (saturation < SATURATION_MAX)
        & (value > VALUE_MIN)
    ).astype(np.uint8) * 255

    # The kernel length is relative to image size.
    kernel_length = max(
        35,
        int(round(minimum_dimension * 0.067)),
    )

    line_thickness = max(
        3,
        int(round(minimum_dimension * 0.0025)),
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_length, line_thickness),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (line_thickness, kernel_length),
    )

    horizontal_lines = cv2.morphologyEx(
        steel_color_mask,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical_lines = cv2.morphologyEx(
        steel_color_mask,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    steel_line_mask = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines,
    )

    return steel_color_mask, steel_line_mask


def detect_line_segments(
    line_mask: np.ndarray,
) -> list[tuple[float, float, tuple[int, int, int, int]]]:
    """
    Detect straight segments using the probabilistic Hough transform.

    Returns:
        [
            (length, angle, (x1, y1, x2, y2)),
            ...
        ]
    """
    height, width = line_mask.shape[:2]
    minimum_dimension = min(height, width)

    edges = cv2.Canny(
        line_mask,
        threshold1=50,
        threshold2=150,
    )

    detected = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=max(
            30,
            int(round(minimum_dimension * 0.045)),
        ),
        minLineLength=max(
            60,
            int(round(minimum_dimension * 0.08)),
        ),
        maxLineGap=max(
            25,
            int(round(minimum_dimension * 0.05)),
        ),
    )

    if detected is None:
        raise RuntimeError(
            "No steel lines were detected. "
            "Try increasing SATURATION_MAX or decreasing VALUE_MIN."
        )

    # Handles both OpenCV return formats:
    # (N, 1, 4) and (N, 4)
    detected = np.asarray(detected, dtype=np.int32).reshape(-1, 4)

    print(f"Detected {len(detected)} Hough line segments")

    segments = []

    for x1, y1, x2, y2 in detected:
        dx = float(x2 - x1)
        dy = float(y2 - y1)

        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx)) % 180.0

        segments.append(
            (
                length,
                angle,
                (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ),
            )
        )

    return segments

def cluster_similar_lines(
    segments: list[
        tuple[float, float, tuple[int, int, int, int]]
    ],
    image_shape: tuple[int, int],
    orientation: str,
) -> list[dict]:
    """
    Groups Hough segments belonging to the same steel bar.

    orientation:
        "vertical" or "horizontal"
    """
    height, width = image_shape
    minimum_dimension = min(height, width)

    center_x = width / 2
    center_y = height / 2

    minimum_length = minimum_dimension * 0.08

    candidates = []

    for length, angle, segment in segments:
        if length < minimum_length:
            continue

        x1, y1, x2, y2 = segment

        if orientation == "vertical":
            if abs(angle - 90) > ANGLE_TOLERANCE:
                continue

            if abs(y2 - y1) < 1:
                continue

            # X coordinate where the segment crosses the image center.
            position = x1 + (
                (center_y - y1)
                * (x2 - x1)
                / (y2 - y1)
            )

        elif orientation == "horizontal":
            horizontal_error = min(
                abs(angle),
                abs(angle - 180),
            )

            if horizontal_error > ANGLE_TOLERANCE:
                continue

            if abs(x2 - x1) < 1:
                continue

            # Y coordinate where the segment crosses the image center.
            position = y1 + (
                (center_x - x1)
                * (y2 - y1)
                / (x2 - x1)
            )

        else:
            raise ValueError(
                "orientation must be 'vertical' or 'horizontal'"
            )

        candidates.append(
            (
                float(position),
                float(length),
                segment,
            )
        )

    candidates.sort(key=lambda item: item[0])

    cluster_tolerance = minimum_dimension * 0.018
    groups: list[list] = []

    for candidate in candidates:
        best_group = None
        best_distance = float("inf")

        for group_index, group in enumerate(groups):
            group_positions = [item[0] for item in group]
            group_weights = [item[1] for item in group]

            group_center = np.average(
                group_positions,
                weights=group_weights,
            )

            distance = abs(candidate[0] - group_center)

            if (
                distance <= cluster_tolerance
                and distance < best_distance
            ):
                best_group = group_index
                best_distance = distance

        if best_group is None:
            groups.append([candidate])
        else:
            groups[best_group].append(candidate)

    clusters = []

    for group in groups:
        positions = [item[0] for item in group]
        weights = [item[1] for item in group]

        clusters.append(
            {
                "position": float(
                    np.average(
                        positions,
                        weights=weights,
                    )
                ),
                "support": float(sum(weights)),
                "segments": [
                    item[2]
                    for item in group
                ],
            }
        )

    return sorted(
        clusters,
        key=lambda cluster: cluster["position"],
    )


def choose_three_grid_lines(
    clusters: list[dict],
    dimension: int,
) -> list[dict]:
    """
    Selects three approximately equally spaced lines:

        left, center, right

    or:

        top, center, bottom
    """
    # Ignore weak clusters when there are many candidates.
    strongest = sorted(
        clusters,
        key=lambda cluster: cluster["support"],
        reverse=True,
    )[:10]

    strongest = sorted(
        strongest,
        key=lambda cluster: cluster["position"],
    )

    best_result = None

    for candidate_triplet in itertools.combinations(
        strongest,
        3,
    ):
        positions = [
            cluster["position"]
            for cluster in candidate_triplet
        ]

        total_span = positions[2] - positions[0]

        if total_span < dimension * 0.18:
            continue

        if total_span > dimension * 0.82:
            continue

        first_spacing = positions[1] - positions[0]
        second_spacing = positions[2] - positions[1]

        if first_spacing <= 0 or second_spacing <= 0:
            continue

        spacing_error = (
            abs(first_spacing - second_spacing)
            / total_span
        )

        if spacing_error > 0.32:
            continue

        edge_penalty = 0.0

        if (
            positions[0] < dimension * 0.03
            or positions[2] > dimension * 0.97
        ):
            edge_penalty = 0.5

        total_support = sum(
            cluster["support"]
            for cluster in candidate_triplet
        )

        score = total_support * (
            1.0
            - 0.75 * spacing_error
            - edge_penalty
        )

        if best_result is None or score > best_result[0]:
            best_result = (
                score,
                candidate_triplet,
            )

    if best_result is None:
        cluster_information = [
            (
                round(cluster["position"], 1),
                round(cluster["support"], 1),
            )
            for cluster in strongest
        ]

        raise RuntimeError(
            "Could not find three equally spaced steel bars. "
            f"Detected clusters: {cluster_information}"
        )

    return list(best_result[1])


def fit_infinite_line(
    cluster: dict,
) -> np.ndarray:
    """
    Fits one line through all Hough segments in a cluster.

    Returns the homogeneous line:

        a*x + b*y + c = 0
    """
    points = []

    for x1, y1, x2, y2 in cluster["segments"]:
        points.append((x1, y1))
        points.append((x2, y2))

    points_array = np.asarray(
        points,
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    vx, vy, x0, y0 = cv2.fitLine(
        points_array,
        cv2.DIST_HUBER,
        0,
        0.01,
        0.01,
    ).reshape(-1)

    a = float(vy)
    b = float(-vx)
    c = -(a * float(x0) + b * float(y0))

    normalization = math.hypot(a, b)

    return np.array(
        [
            a / normalization,
            b / normalization,
            c / normalization,
        ],
        dtype=np.float64,
    )


def line_intersection(
    first_line: np.ndarray,
    second_line: np.ndarray,
) -> np.ndarray:
    intersection = np.cross(
        first_line,
        second_line,
    )

    if abs(intersection[2]) < 1e-8:
        raise RuntimeError(
            "Detected steel lines are parallel or invalid."
        )

    return np.array(
        [
            intersection[0] / intersection[2],
            intersection[1] / intersection[2],
        ],
        dtype=np.float32,
    )


def detect_grid(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        grid_points:
            A 3x3 array of steel-bar intersection coordinates
            in the original image.

        detection_image:
            Resized image used for detection.

        steel_line_mask:
            Detected linear steel structures.
    """
    detection_image, scale = resize_for_detection(
        image,
        MAX_DETECTION_DIMENSION,
    )

    _, steel_line_mask = detect_possible_steel(
        detection_image
    )

    segments = detect_line_segments(
        steel_line_mask
    )

    vertical_clusters = cluster_similar_lines(
        segments,
        steel_line_mask.shape,
        orientation="vertical",
    )

    horizontal_clusters = cluster_similar_lines(
        segments,
        steel_line_mask.shape,
        orientation="horizontal",
    )

    selected_vertical = choose_three_grid_lines(
        vertical_clusters,
        detection_image.shape[1],
    )

    selected_horizontal = choose_three_grid_lines(
        horizontal_clusters,
        detection_image.shape[0],
    )

    vertical_lines = [
        fit_infinite_line(cluster)
        for cluster in selected_vertical
    ]

    horizontal_lines = [
        fit_infinite_line(cluster)
        for cluster in selected_horizontal
    ]

    grid_points = np.zeros(
        (3, 3, 2),
        dtype=np.float32,
    )

    for row, horizontal_line in enumerate(
        horizontal_lines
    ):
        for column, vertical_line in enumerate(
            vertical_lines
        ):
            point = line_intersection(
                horizontal_line,
                vertical_line,
            )

            # Convert coordinates back to original image size.
            grid_points[row, column] = point / scale

    return (
        grid_points,
        detection_image,
        steel_line_mask,
    )


def warp_cell(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """
    Perspective-corrects one cell and removes its steel border.
    """
    canvas_size = int(
        round(
            CELL_SIZE
            / (1.0 - 2.0 * INNER_MARGIN_FRACTION)
        )
    )

    destination = np.float32(
        [
            [0, 0],
            [canvas_size - 1, 0],
            [canvas_size - 1, canvas_size - 1],
            [0, canvas_size - 1],
        ]
    )

    transformation = cv2.getPerspectiveTransform(
        corners.astype(np.float32),
        destination,
    )

    warped = cv2.warpPerspective(
        image,
        transformation,
        (canvas_size, canvas_size),
        flags=cv2.INTER_LINEAR,
    )

    margin = (canvas_size - CELL_SIZE) // 2

    cropped = warped[
        margin:margin + CELL_SIZE,
        margin:margin + CELL_SIZE,
    ]

    return cropped


def save_debug_grid(
    image: np.ndarray,
    grid_points: np.ndarray,
    output_path: Path,
) -> None:
    preview = image.copy()

    line_color = (0, 0, 255)
    point_color = (0, 255, 255)

    # Draw horizontal steel bars.
    for row in range(3):
        points = grid_points[row].astype(np.int32)

        cv2.polylines(
            preview,
            [points.reshape(-1, 1, 2)],
            isClosed=False,
            color=line_color,
            thickness=8,
        )

    # Draw vertical steel bars.
    for column in range(3):
        points = grid_points[:, column].astype(np.int32)

        cv2.polylines(
            preview,
            [points.reshape(-1, 1, 2)],
            isClosed=False,
            color=line_color,
            thickness=8,
        )

    for row in range(3):
        for column in range(3):
            x, y = grid_points[row, column].astype(int)

            cv2.circle(
                preview,
                (x, y),
                radius=14,
                color=point_color,
                thickness=-1,
            )

    if not cv2.imwrite(
        str(output_path),
        preview,
    ):
        raise IOError(
            f"Could not save debug image: {output_path}"
        )


def save_image(
    path: Path,
    image: np.ndarray,
) -> None:
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not save: {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    grid_points, _, steel_line_mask = detect_grid(
        image
    )

    save_debug_grid(
        image,
        grid_points,
        OUTPUT_DIR / "debug_detected_grid.jpg",
    )

    save_image(
        OUTPUT_DIR / "debug_steel_lines.png",
        steel_line_mask,
    )

    cell_names = {
        (0, 0): "01_top_left.jpg",
        (0, 1): "02_top_right.jpg",
        (1, 0): "03_bottom_left.jpg",
        (1, 1): "04_bottom_right.jpg",
    }

    for row in range(2):
        for column in range(2):
            corners = np.float32(
                [
                    grid_points[row, column],
                    grid_points[row, column + 1],
                    grid_points[row + 1, column + 1],
                    grid_points[row + 1, column],
                ]
            )

            cell = warp_cell(
                image,
                corners,
            )

            output_path = (
                OUTPUT_DIR
                / cell_names[(row, column)]
            )

            save_image(
                output_path,
                cell,
            )

            print(f"Saved: {output_path}")

    print(
        f"\nFinished. Results are in: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()