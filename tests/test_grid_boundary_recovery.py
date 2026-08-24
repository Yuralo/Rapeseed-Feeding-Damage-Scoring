import importlib.util
import unittest


GRID_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module) is not None for module in ("cv2", "numpy")
)


def _cluster(position, support, angle=0.0):
    return {
        "position": float(position),
        "support": float(support),
        "angle": float(angle),
        "segments": [(int(position), 100, int(position), 900)],
    }


@unittest.skipUnless(GRID_DEPENDENCIES_AVAILABLE, "OpenCV grid dependencies are unavailable")
class GridBoundaryRecoveryTests(unittest.TestCase):
    def test_recovers_missing_right_bar_from_two_strong_adjacent_bars(self):
        from utils.image import choose_three_grid_lines

        clusters = [
            _cluster(779, 151),
            _cluster(814, 1188, -1.0),
            _cluster(1037, 98),
            _cluster(1100, 1780, 1.0),
        ]

        selected = choose_three_grid_lines(
            clusters,
            1600,
            orientation="vertical",
            allow_boundary_recovery=True,
        )

        self.assertEqual([round(line["position"]) for line in selected], [814, 1100, 1386])
        self.assertTrue(selected[-1]["inferred"])

    def test_does_not_extrapolate_a_missing_line_in_the_image_interior(self):
        from utils.image import choose_three_grid_lines

        clusters = [_cluster(500, 1000), _cluster(700, 1200)]

        with self.assertRaisesRegex(RuntimeError, "Could not find three equally spaced"):
            choose_three_grid_lines(
                clusters,
                1600,
                orientation="vertical",
                allow_boundary_recovery=True,
            )

    def test_normal_three_line_solution_is_preferred_over_recovery(self):
        from utils.image import choose_three_grid_lines

        clusters = [_cluster(250, 900), _cluster(600, 1100), _cluster(950, 1000)]

        selected = choose_three_grid_lines(
            clusters,
            1600,
            orientation="vertical",
            allow_boundary_recovery=True,
        )

        self.assertEqual([line["position"] for line in selected], [250.0, 600.0, 950.0])
        self.assertFalse(any(line.get("inferred", False) for line in selected))


if __name__ == "__main__":
    unittest.main()
