import tempfile
import unittest
from pathlib import Path

from e_emotion.config import ConfigError, load_config
from e_emotion.problem2_fair.registry import load_registry


class ConfigTests(unittest.TestCase):
    def test_base_config_defers_variant_selection(self) -> None:
        config = load_config(Path("configs/base.yaml"))
        self.assertIsNone(config.variant)
        self.assertTrue(config.data_root.parts[-2:] == ("data", "raw"))
        self.assertEqual(config.feature_version, "aligned_50")
        self.assertEqual(dict(config.split_files), {"train": "train.npz", "valid": "valid.npz", "test": "test.npz"})
        self.assertEqual(config.source_format, "npz")

    def test_config_rejects_data_root_outside_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                "dataset:\n  data_root: ../outside\n  variant: aligned\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_method_registry_records_mult_both_supported_views(self) -> None:
        mult = load_registry().get("mult")
        self.assertEqual(mult.input_layout, "native")
        self.assertEqual(mult.views, ("aligned_po", "unaligned_po"))


if __name__ == "__main__":
    unittest.main()
