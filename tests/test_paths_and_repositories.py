import tempfile
import unittest
from pathlib import Path

from e_emotion.data import DataPathPolicy, MissingModalityRepository


class PathAndRepositoryTests(unittest.TestCase):
    def test_path_policy_blocks_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            root.mkdir()
            policy = DataPathPolicy(root)
            with self.assertRaises(ValueError):
                policy.resolve(Path("..") / "secret.pkl")

    def test_missing_repository_uses_variant_specific_names(self) -> None:
        repository = MissingModalityRepository(Path("data/raw"))
        self.assertTrue(repository.path_for(1, "aligned").name == "附件3_01.pkl")
        self.assertTrue(repository.path_for(1, "unaligned").name == "附件3_未对齐版本_01.pkl")


if __name__ == "__main__":
    unittest.main()
