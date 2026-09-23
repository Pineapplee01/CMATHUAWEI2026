import unittest

import numpy as np

from e_emotion.evaluation import accuracy, f1_macro, mae, pearson


class MetricTests(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        truth = np.array([0, 1, 2, 2])
        prediction = np.array([0, 2, 2, 1])
        self.assertAlmostEqual(accuracy(truth, prediction), 0.5)
        self.assertAlmostEqual(f1_macro(truth, prediction), (1.0 + 0.0 + 0.5) / 3)

    def test_regression_metrics(self) -> None:
        truth = np.array([0.0, 1.0, 2.0])
        prediction = np.array([0.0, 2.0, 1.0])
        self.assertAlmostEqual(mae(truth, prediction), 2.0 / 3.0)
        self.assertAlmostEqual(pearson(truth, prediction), 0.5)


if __name__ == "__main__":
    unittest.main()
