"""Shared competition scores with backward-compatible paper-diagnostic aliases."""
import numpy as np

from e_emotion.evaluation import score_predictions
from e_emotion.evaluation.metrics import f1_scores
from e_emotion.evaluation.outputs import polarity, select_neutral_threshold


def metrics(truth, prediction, threshold=0.0):
    """Input must already be final intensity; this wrapper never clips it."""
    predicted = polarity(prediction, threshold)  # Frozen AUMDF head rule, not threshold fitting.
    competition = score_predictions(truth, prediction, predicted)
    y, p = np.asarray(truth), np.asarray(prediction)
    nonzero = y != 0
    diagnostics = {
        "accuracy_7": float((np.round(y) == np.round(p)).mean()),
        "accuracy_2_nonzero": float(((y[nonzero]>0) == (p[nonzero]>0)).mean()) if nonzero.any() else None,
        "weighted_f1_2_nonzero": f1_scores(y[nonzero]>0, p[nonzero]>0, (False,True))[1] if nonzero.any() else None,
    }
    return {
        "protocol_version": competition["protocol_version"], "competition": competition,
        "paper_diagnostics": diagnostics, "n": competition["n"], "mae": competition["mae"],
        "pearson": competition["pearson"], "accuracy_3": competition["accuracy"],
        "macro_f1_3": competition["macro_f1"], "weighted_f1_3": competition["weighted_f1"],
        "neutral_threshold": float(threshold), **diagnostics,
    }
