import tempfile
import unittest
from pathlib import Path

from e_emotion.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_base_config_defers_variant_selection(self) -> None:
        config = load_config(Path("configs/base.yaml"))
        self.assertIsNone(config.variant)
        self.assertTrue(config.data_root.parts[-2:] == ("data", "raw"))

    def test_config_rejects_data_root_outside_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                "dataset:\n  data_root: ../outside\n  variant: aligned\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
