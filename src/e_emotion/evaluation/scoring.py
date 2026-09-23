"""Pure scoring of final predictions. No clipping, threshold search or calibration."""
import numpy as np

from e_emotion.evaluation.metrics import accuracy, f1_scores, mae, pearson
from e_emotion.evaluation.outputs import PROTOCOL_VERSION, classes, intensities, polarity


def score_predictions(truth_intensity, prediction_intensity, prediction_polarity, *,
                      truth_polarity=None) -> dict:
    actual = intensities(truth_intensity, "truth intensity")
    predicted = intensities(prediction_intensity, "final prediction intensity")
    actual_classes = polarity(actual, 0)
    if truth_polarity is not None and not np.array_equal(classes(truth_polarity), actual_classes):
        raise ValueError("truth polarity disagrees with official intensity label mapping")
    predicted_classes = classes(prediction_polarity)
    if not (actual.shape == predicted.shape == predicted_classes.shape):
        raise ValueError("truth and prediction must have the same shape")
    macro, weighted = f1_scores(actual_classes, predicted_classes)
    correlation = pearson(actual, predicted)
    reason = None
    if actual.size < 2:
        reason = "fewer_than_two_samples"
    elif np.all(actual == actual[0]):
        reason = "constant_truth"
    elif np.all(predicted == predicted[0]):
        reason = "constant_prediction"
    elif correlation is None:
        reason = "numerically_zero_variance"
    return {
        "protocol_version": PROTOCOL_VERSION, "n": int(actual.size),
        "accuracy": accuracy(actual_classes, predicted_classes), "macro_f1": macro,
        "weighted_f1": weighted, "mae": mae(actual, predicted), "pearson": correlation,
        "pearson_undefined_reason": reason, "class_order": ["Negative", "Neutral", "Positive"],
        "class_support": [int(np.sum(actual_classes == label)) for label in range(3)],
        "confusion_matrix": [[int(np.sum((actual_classes == a) & (predicted_classes == p)))
                              for p in range(3)] for a in range(3)],
    }
