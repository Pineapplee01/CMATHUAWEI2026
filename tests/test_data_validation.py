import unittest

import numpy as np

from e_emotion.data.validation import validate_attachment2_payload


class DataValidationTests(unittest.TestCase):
    def test_valid_aligned_payload_reports_split_sizes(self) -> None:
        payload = {
            "train": {
                "id": ["v$_$1"],
                "text": np.zeros((1, 50, 768), dtype=np.float32),
                "audio": np.zeros((1, 50, 74), dtype=np.float64),
                "vision": np.zeros((1, 50, 35), dtype=np.float64),
                "classification_labels": np.array([0.0]),
                "regression_labels": np.array([-1.0]),
            },
            "valid": {
                "id": ["v$_$2"],
                "text": np.zeros((1, 50, 768), dtype=np.float32),
                "audio": np.zeros((1, 50, 74), dtype=np.float64),
                "vision": np.zeros((1, 50, 35), dtype=np.float64),
                "classification_labels": np.array([2.0]),
                "regression_labels": np.array([1.0]),
            },
            "test": {
                "id": ["v$_$3"],
                "text": np.zeros((1, 50, 768), dtype=np.float32),
                "audio": np.zeros((1, 50, 74), dtype=np.float64),
                "vision": np.zeros((1, 50, 35), dtype=np.float64),
                "classification_labels": np.array([1.0]),
                "regression_labels": np.array([0.0]),
            },
        }
        report = validate_attachment2_payload(payload, variant="aligned")
        self.assertEqual(report.split_sizes, {"train": 1, "valid": 1, "test": 1})
        self.assertEqual(report.errors, ())

    def test_invalid_label_range_is_reported(self) -> None:
        payload = {
            "train": {
                "id": ["v$_$1"],
                "text": np.zeros((1, 50, 768), dtype=np.float32),
                "audio": np.zeros((1, 50, 74), dtype=np.float64),
                "vision": np.zeros((1, 50, 35), dtype=np.float64),
                "classification_labels": np.array([4.0]),
                "regression_labels": np.array([0.0]),
            }
        }
        report = validate_attachment2_payload(payload, variant="aligned")
        self.assertTrue(any("classification_labels" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
