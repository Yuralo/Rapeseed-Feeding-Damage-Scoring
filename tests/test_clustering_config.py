import unittest

from analysis.clustering.config import load_analysis_config


class ClusteringConfigTests(unittest.TestCase):
    def test_default_comparison_has_unique_explicit_stages(self):
        analysis, representations = load_analysis_config(
            "analysis/clustering/config.toml"
        )
        self.assertEqual(analysis.seed, 42)
        self.assertEqual(len(representations), 4)
        self.assertEqual(len({item.name for item in representations}), 4)
        self.assertEqual(
            {item.feature for item in representations}, {"backbone", "head"}
        )


if __name__ == "__main__":
    unittest.main()
