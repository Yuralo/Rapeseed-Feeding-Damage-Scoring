import json
import tempfile
import unittest
from pathlib import Path

from rapeseed_damage.artifacts import git_state, write_json


class ArtifactTests(unittest.TestCase):
    def test_write_json_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "run" / "summary.json"
            write_json(destination, {"score": 1.25})
            self.assertEqual(json.loads(destination.read_text()), {"score": 1.25})

    def test_git_state_records_current_repository(self):
        state = git_state(Path(__file__).resolve().parents[1])
        self.assertEqual(len(state["commit"]), 40)
        self.assertIsInstance(state["dirty"], bool)


if __name__ == "__main__":
    unittest.main()

