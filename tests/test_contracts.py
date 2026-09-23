import unittest

import numpy as np

from e_emotion.contracts import (
    ModalitySequence,
    Polarity,
    SampleId,
)


class ContractTests(unittest.TestCase):
    def test_sample_id_uses_mosei_serialization(self) -> None:
        sample_id = SampleId(video_id="-3g5yACwYnA", clip_id="4")
        self.assertEqual(sample_id.as_string, "-3g5yACwYnA$_$4")

    def test_polarity_accepts_numeric_and_text_labels(self) -> None:
        self.assertIs(Polarity.from_value(0), Polarity.NEGATIVE)
        self.assertIs(Polarity.from_value(1.0), Polarity.NEUTRAL)
        self.assertIs(Polarity.from_value("Positive"), Polarity.POSITIVE)

    def test_modality_sequence_rejects_mask_with_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            ModalitySequence(
                values=np.zeros((3, 2)),
                valid_length=3,
                mask=np.ones(2, dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()
