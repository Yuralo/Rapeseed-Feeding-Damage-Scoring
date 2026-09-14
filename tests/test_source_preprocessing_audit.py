import unittest

from experiments.dinov3_mixed_domain_adaptation.inspect_sources import (
    family_aware_random_sample,
    filename_family,
)


class SourcePreprocessingAuditTests(unittest.TestCase):
    def test_filename_family(self):
        self.assertEqual(filename_family("IMG_2533.JPG"), "IMG")
        self.assertEqual(filename_family("20251021_153843.jpg"), "timestamp")
        self.assertEqual(filename_family("capture-a.jpg"), "other")

    def test_sample_is_reproducible_and_covers_both_common_families(self):
        rows = [
            {"file_name": f"IMG_{index:04d}.jpg", "value": f"img-{index}"}
            for index in range(10)
        ] + [
            {"file_name": f"20251021_{index:06d}.jpg", "value": f"time-{index}"}
            for index in range(10)
        ]

        first = family_aware_random_sample(
            rows,
            6,
            seed=42,
            group_key="cohort/source",
            filename_column="file_name",
        )
        second = family_aware_random_sample(
            rows,
            6,
            seed=42,
            group_key="cohort/source",
            filename_column="file_name",
        )

        self.assertEqual(first, second)
        self.assertEqual({filename_family(row["file_name"]) for row in first}, {"IMG", "timestamp"})
        self.assertEqual(len(first), 6)


if __name__ == "__main__":
    unittest.main()
