import unittest

from e_emotion.alignment import FeatureExtractor, TemporalAligner
from e_emotion.explainability import EvidenceMapper, ExplainablePredictor
from e_emotion.robustness import MissingnessSimulator, RobustPredictor


class InterfaceTests(unittest.TestCase):
    def test_task_protocols_are_importable(self) -> None:
        for interface in (
            FeatureExtractor,
            TemporalAligner,
            MissingnessSimulator,
            RobustPredictor,
            ExplainablePredictor,
            EvidenceMapper,
        ):
            self.assertIsNotNone(interface)


if __name__ == "__main__":
    unittest.main()
