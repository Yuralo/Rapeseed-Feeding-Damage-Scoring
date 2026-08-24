import cv2
import numpy as np
import math
import itertools

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

# Boundary recovery is deliberately narrower than normal grid detection. It is
# used only after exact three-line selection fails and only extrapolates one
# missing outer bar from two strong adjacent bars.
BOUNDARY_RECOVERY_MIN_POSITION = -0.15
BOUNDARY_RECOVERY_MAX_POSITION = 1.15
BOUNDARY_RECOVERY_EDGE_FRACTION = 0.18
BOUNDARY_RECOVERY_MAX_ANGLE_DIFFERENCE = 8.0


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
    """Build a permissive steel mask and an edge mask for Hough fallback."""
    minimum_dimension = min(image.shape[:2])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    steel_color_mask = (
        (saturation < SATURATION_MAX)
        & (value > VALUE_MIN)
    ).astype(np.uint8) * 255

    close_size = max(3, int(round(minimum_dimension * 0.004)))
    if close_size % 2 == 0:
        close_size += 1
    steel_color_mask = cv2.morphologyEx(
        steel_color_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median_intensity = float(np.median(gray))
    lower = int(max(20, 0.55 * median_intensity))
    upper = int(min(220, max(lower + 30, 1.45 * median_intensity)))
    edges = cv2.Canny(gray, lower, upper)

    support_size = max(3, int(round(minimum_dimension * 0.007)))
    if support_size % 2 == 0:
        support_size += 1
    color_support = cv2.dilate(
        steel_color_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support_size, support_size)),
    )
    steel_line_mask = cv2.bitwise_and(edges, color_support)
    return steel_color_mask, steel_line_mask


def _segments_from_lsd(
    source_image: np.ndarray,
) -> list[tuple[float, float, tuple[int, int, int, int]]]:
    """Detect long coherent segments without assuming exact axis alignment."""
    if source_image.ndim == 3:
        gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = source_image.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    minimum_dimension = min(gray.shape[:2])
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    detected = detector.detect(gray)[0]
    if detected is None:
        return []

    minimum_length = max(35.0, minimum_dimension * 0.04)
    segments = []
    for x1, y1, x2, y2 in np.asarray(detected).reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < minimum_length:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        segments.append(
            (
                length,
                angle,
                (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                ),
            )
        )
    return segments


def _segments_from_hough(
    line_mask: np.ndarray,
) -> list[tuple[float, float, tuple[int, int, int, int]]]:
    """Compatibility fallback for images where LSD returns too few segments."""
    height, width = line_mask.shape[:2]
    minimum_dimension = min(height, width)
    detected = cv2.HoughLinesP(
        line_mask,
        rho=1,
        theta=np.pi / 720,
        threshold=max(25, int(round(minimum_dimension * 0.03))),
        minLineLength=max(40, int(round(minimum_dimension * 0.05))),
        maxLineGap=max(20, int(round(minimum_dimension * 0.04))),
    )
    if detected is None:
        return []
    segments = []
    for x1, y1, x2, y2 in np.asarray(detected).reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        segments.append(
            (length, angle, (int(x1), int(y1), int(x2), int(y2)))
        )
    return segments


def detect_line_segments(
    line_mask: np.ndarray,
    source_image: np.ndarray | None = None,
) -> list[tuple[float, float, tuple[int, int, int, int]]]:
    """Use the notebook's stronger LSD path, with Hough as fallback."""
    if source_image is not None:
        segments = _segments_from_lsd(source_image)
        if segments:
            return segments
    segments = _segments_from_hough(line_mask)
    if not segments:
        raise RuntimeError("No sufficiently long steel-line candidates were detected.")
    return segments


def _signed_orientation_error(angle: float, orientation: str) -> float:
    if orientation == "horizontal":
        return angle if angle <= 90.0 else angle - 180.0
    if orientation == "vertical":
        return angle - 90.0
    raise ValueError("orientation must be 'vertical' or 'horizontal'")

def cluster_similar_lines(
    segments: list[tuple[float, float, tuple[int, int, int, int]]],
    image_shape: tuple[int, int],
    orientation: str,
) -> list[dict]:
    """Group fragments belonging to the same projected steel bar."""
    height, width = image_shape
    minimum_dimension = min(height, width)
    center_x, center_y = width / 2.0, height / 2.0
    candidates = []
    for length, angle, segment in segments:
        orientation_error = _signed_orientation_error(angle, orientation)
        if abs(orientation_error) > ANGLE_TOLERANCE:
            continue
        x1, y1, x2, y2 = segment
        if orientation == "vertical":
            if abs(y2 - y1) < 1:
                continue
            position = x1 + (center_y - y1) * (x2 - x1) / (y2 - y1)
        else:
            if abs(x2 - x1) < 1:
                continue
            position = y1 + (center_x - x1) * (y2 - y1) / (x2 - x1)

        relevant_dimension = width if orientation == "vertical" else height
        if not -0.15 * relevant_dimension <= position <= 1.15 * relevant_dimension:
            continue
        candidates.append(
            (float(position), float(length), segment, float(orientation_error))
        )
    candidates.sort(key=lambda item: item[0])
    position_tolerance = minimum_dimension * 0.018
    angle_tolerance = max(4.0, ANGLE_TOLERANCE * 0.6)
    groups: list[list] = []
    for candidate in candidates:
        best_group = None
        best_cost = float("inf")
        for group_index, group in enumerate(groups):
            weights = [item[1] for item in group]
            group_position = float(
                np.average([item[0] for item in group], weights=weights)
            )
            group_angle = float(
                np.average([item[3] for item in group], weights=weights)
            )
            position_distance = abs(candidate[0] - group_position)
            angle_distance = abs(candidate[3] - group_angle)
            if (
                position_distance <= position_tolerance
                and angle_distance <= angle_tolerance
            ):
                cost = (
                    position_distance / position_tolerance
                    + angle_distance / angle_tolerance
                )
                if cost < best_cost:
                    best_group = group_index
                    best_cost = cost
        if best_group is None:
            groups.append([candidate])
        else:
            groups[best_group].append(candidate)
    clusters = []
    for group in groups:
        weights = [item[1] for item in group]
        clusters.append(
            {
                "position": float(np.average([item[0] for item in group], weights=weights)),
                "support": float(sum(weights)),
                "angle": float(np.average([item[3] for item in group], weights=weights)),
                "segments": [item[2] for item in group],
            }
        )
    return sorted(clusters, key=lambda cluster: cluster["position"])


def choose_three_grid_lines(
    clusters: list[dict],
    dimension: int,
    *,
    orientation: str,
    allow_boundary_recovery: bool = False,
) -> list[dict]:
    """Select three strong, approximately equally spaced projected bars."""
    strongest = sorted(
        clusters,
        key=lambda cluster: cluster["support"],
        reverse=True,
    )[:12]
    strongest.sort(key=lambda cluster: cluster["position"])
    best_result = None
    for candidate_triplet in itertools.combinations(strongest, 3):
        positions = [cluster["position"] for cluster in candidate_triplet]
        total_span = positions[2] - positions[0]
        if not dimension * 0.18 <= total_span <= dimension * 0.82:
            continue
        first_spacing = positions[1] - positions[0]
        second_spacing = positions[2] - positions[1]
        if first_spacing <= 0 or second_spacing <= 0:
            continue
        spacing_error = abs(first_spacing - second_spacing) / total_span
        if spacing_error > 0.32:
            continue
        total_support = sum(cluster["support"] for cluster in candidate_triplet)
        mean_angle = float(
            np.average(
                [cluster.get("angle", 0.0) for cluster in candidate_triplet],
                weights=[cluster["support"] for cluster in candidate_triplet],
            )
        )
        angle_spread = max(
            abs(cluster.get("angle", 0.0) - mean_angle)
            for cluster in candidate_triplet
        )
        score = total_support * math.exp(-4.0 * spacing_error - 0.08 * angle_spread)
        if best_result is None or score > best_result[0]:
            best_result = (score, candidate_triplet)
    if best_result is None:
        if allow_boundary_recovery:
            recovered = _recover_boundary_grid_line(
                strongest,
                dimension,
                orientation=orientation,
            )
            if recovered is not None:
                return recovered
        cluster_information = [
            (
                round(cluster["position"], 1),
                round(cluster["support"], 1),
                round(cluster.get("angle", 0.0), 1),
            )
            for cluster in strongest
        ]
        raise RuntimeError(
            "Could not find three equally spaced steel bars. "
            "Detected clusters (position, support, angle): "
            f"{cluster_information}"
        )
    return list(best_result[1])


def _translated_cluster(
    source: dict,
    offset: float,
    *,
    orientation: str,
) -> dict:
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")
    dx = offset if orientation == "vertical" else 0.0
    dy = offset if orientation == "horizontal" else 0.0
    translated_segments = [
        (
            int(round(x1 + dx)),
            int(round(y1 + dy)),
            int(round(x2 + dx)),
            int(round(y2 + dy)),
        )
        for x1, y1, x2, y2 in source["segments"]
    ]
    return {
        "position": float(source["position"] + offset),
        "support": float(source["support"] * 0.25),
        "angle": float(source.get("angle", 0.0)),
        "segments": translated_segments,
        "inferred": True,
        "inferred_from_position": float(source["position"]),
    }


def _recover_boundary_grid_line(
    clusters: list[dict],
    dimension: int,
    *,
    orientation: str,
) -> list[dict] | None:
    """Infer one edge-clipped outer bar from two strong adjacent grid bars."""
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    minimum_support = dimension * 0.08
    best_result = None
    for first, second in itertools.combinations(clusters, 2):
        spacing = float(second["position"] - first["position"])
        if not dimension * 0.09 <= spacing <= dimension * 0.41:
            continue
        first_support = float(first["support"])
        second_support = float(second["support"])
        if min(first_support, second_support) < minimum_support:
            continue
        if min(first_support, second_support) / max(first_support, second_support) < 0.15:
            continue
        has_supported_line_between = any(
            first["position"] < cluster["position"] < second["position"]
            and float(cluster["support"]) >= minimum_support
            for cluster in clusters
        )
        if has_supported_line_between:
            continue
        angle_difference = abs(
            float(first.get("angle", 0.0)) - float(second.get("angle", 0.0))
        )
        if angle_difference > BOUNDARY_RECOVERY_MAX_ANGLE_DIFFERENCE:
            continue

        inferred_candidates = (
            (float(first["position"] - spacing), first, -spacing, "before"),
            (float(second["position"] + spacing), second, spacing, "after"),
        )
        for inferred_position, source, offset, side in inferred_candidates:
            normalized_position = inferred_position / dimension
            if not (
                BOUNDARY_RECOVERY_MIN_POSITION
                <= normalized_position
                <= BOUNDARY_RECOVERY_MAX_POSITION
            ):
                continue
            if side == "before":
                if normalized_position > BOUNDARY_RECOVERY_EDGE_FRACTION:
                    continue
                boundary_distance = abs(normalized_position)
            else:
                if normalized_position < 1.0 - BOUNDARY_RECOVERY_EDGE_FRACTION:
                    continue
                boundary_distance = abs(1.0 - normalized_position)

            support = first_support + second_support
            score = support * math.exp(
                -0.15 * angle_difference - 2.0 * boundary_distance
            )
            inferred = _translated_cluster(source, offset, orientation=orientation)
            lines = [first, second, inferred]
            lines.sort(key=lambda cluster: cluster["position"])
            if best_result is None or score > best_result[0]:
                best_result = (score, lines)
    return None if best_result is None else best_result[1]


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


def _select_grid_lines(
    segments: list[tuple[float, float, tuple[int, int, int, int]]],
    image_shape: tuple[int, int],
    *,
    allow_boundary_recovery: bool = False,
) -> tuple[list[dict], list[dict]]:
    vertical_clusters = cluster_similar_lines(
        segments, image_shape, orientation="vertical"
    )
    horizontal_clusters = cluster_similar_lines(
        segments, image_shape, orientation="horizontal"
    )
    return (
        choose_three_grid_lines(
            vertical_clusters,
            image_shape[1],
            orientation="vertical",
            allow_boundary_recovery=allow_boundary_recovery,
        ),
        choose_three_grid_lines(
            horizontal_clusters,
            image_shape[0],
            orientation="horizontal",
            allow_boundary_recovery=allow_boundary_recovery,
        ),
    )


def _render_grid_mask(
    image_shape: tuple[int, int],
    grid_points: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    thickness = max(2, int(round(min(image_shape) * 0.005)))
    for row in range(3):
        start = tuple(np.round(grid_points[row, 0]).astype(int))
        end = tuple(np.round(grid_points[row, 2]).astype(int))
        cv2.line(mask, start, end, 255, thickness)
    for column in range(3):
        start = tuple(np.round(grid_points[0, column]).astype(int))
        end = tuple(np.round(grid_points[2, column]).astype(int))
        cv2.line(mask, start, end, 255, thickness)
    return mask


def detect_grid(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect the 3x3 intersections using the notebook's LSD/Hough strategy."""
    if image is None or image.ndim != 3:
        raise ValueError("detect_grid expects a valid BGR color image")
    detection_image, scale = resize_for_detection(
        image,
        MAX_DETECTION_DIMENSION,
    )
    _, candidate_mask = detect_possible_steel(detection_image)
    segments = detect_line_segments(
        candidate_mask,
        source_image=detection_image,
    )

    try:
        selected_vertical, selected_horizontal = _select_grid_lines(
            segments, candidate_mask.shape
        )
    except RuntimeError as primary_error:
        hough_segments = _segments_from_hough(candidate_mask)
        try:
            selected_vertical, selected_horizontal = _select_grid_lines(
                segments + hough_segments,
                candidate_mask.shape,
            )
        except RuntimeError as fallback_error:
            try:
                selected_vertical, selected_horizontal = _select_grid_lines(
                    segments + hough_segments,
                    candidate_mask.shape,
                    allow_boundary_recovery=True,
                )
            except RuntimeError as recovery_error:
                raise RuntimeError(
                    "Grid detection failed with LSD, Hough, and conservative "
                    "boundary recovery. "
                    f"LSD result: {primary_error}; Hough fallback: {fallback_error}; "
                    f"boundary recovery: {recovery_error}"
                ) from recovery_error

    vertical_lines = [
        fit_infinite_line(cluster) for cluster in selected_vertical
    ]
    horizontal_lines = [
        fit_infinite_line(cluster) for cluster in selected_horizontal
    ]
    detection_grid_points = np.zeros((3, 3, 2), dtype=np.float32)
    for row, horizontal_line in enumerate(horizontal_lines):
        for column, vertical_line in enumerate(vertical_lines):
            detection_grid_points[row, column] = line_intersection(
                horizontal_line, vertical_line
            )

    if not np.isfinite(detection_grid_points).all():
        raise RuntimeError("Grid intersections contain invalid coordinates.")
    outer_corners = np.float32(
        [
            detection_grid_points[0, 0],
            detection_grid_points[0, 2],
            detection_grid_points[2, 2],
            detection_grid_points[2, 0],
        ]
    )
    height, width = candidate_mask.shape
    coordinate_margin = 0.25
    if (
        outer_corners[:, 0].min() < -coordinate_margin * width
        or outer_corners[:, 0].max() > (1.0 + coordinate_margin) * width
        or outer_corners[:, 1].min() < -coordinate_margin * height
        or outer_corners[:, 1].max() > (1.0 + coordinate_margin) * height
    ):
        raise RuntimeError(
            "Detected or inferred grid corners extend implausibly far outside the image."
        )
    grid_area = abs(cv2.contourArea(outer_corners))
    if grid_area < 0.025 * candidate_mask.size:
        raise RuntimeError(
            "Detected lines form an implausibly small grid; refusing the result."
        )
    steel_line_mask = _render_grid_mask(
        candidate_mask.shape, detection_grid_points
    )
    return detection_grid_points / scale, detection_image, steel_line_mask


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

def warp_big_square(
    image: np.ndarray,
    grid_points: np.ndarray,
    size: int = 1400,
    inner_margin_fraction: float = 0.0,
) -> np.ndarray:
    """
    Extract and perspective-correct the entire large quadrat.

    Uses only the four outer corners of the detected steel grid. A positive
    inner margin maps an inset region directly to the output canvas, removing
    border artifacts without first resampling an intermediate crop.
    """

    if not 0.0 <= inner_margin_fraction < 0.5:
        raise ValueError("inner_margin_fraction must be in [0, 0.5)")

    corners = np.float32([
        grid_points[0, 0],  # top-left
        grid_points[0, 2],  # top-right
        grid_points[2, 2],  # bottom-right
        grid_points[2, 0],  # bottom-left
    ])

    output_extent = size - 1
    offset = output_extent * inner_margin_fraction / (1.0 - 2.0 * inner_margin_fraction)
    destination = np.float32([
        [-offset, -offset],
        [output_extent + offset, -offset],
        [output_extent + offset, output_extent + offset],
        [-offset, output_extent + offset],
    ])

    matrix = cv2.getPerspectiveTransform(
        corners,
        destination,
    )

    big_square = cv2.warpPerspective(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
    )

    return big_square


def image_to_ndarray(image: str) -> np.ndarray:
    return cv2.imread(str(image))


def to_rgb(array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
