"""Evaluate EF-LSTM on the common canonical 63-condition Q2 matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model import MaskAwareEFLSTM
from runtime import COMBINATIONS, EXPECTED_MASK_SHA256, FRACTIONS, POSITIONS, ROOT, load_bundle, load_common_q2_index, predict, q2_condition, save_json, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-name", default="q2_strict_test_maskaware_v1")
    args = parser.parse_args()
    if Path(args.run).name != args.run or Path(args.output_name).name != args.output_name:
        raise ValueError("run and output names must be directory names")
    run = ROOT / "runs" / args.run
    output = run / args.output_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads((run / "q2_mask_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mask_sha256") != EXPECTED_MASK_SHA256:
        raise RuntimeError("run is not bound to the canonical Q2 mask manifest")
    threshold = json.loads((run / "neutral_interval.json").read_text(encoding="utf-8"))
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    model = MaskAwareEFLSTM(**{key: config[key] for key in ("text_dim", "audio_dim", "vision_dim", "hidden_dim", "num_layers", "dropout")})
    device = torch.device(args.device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    bundle = load_bundle("test")
    index = load_common_q2_index()
    output.mkdir()
    conditions = []
    rows = []
    for combination in COMBINATIONS:
        for position in POSITIONS:
            for fraction in FRACTIONS:
                condition, removed, eligible = q2_condition(bundle, index, combination, position, fraction)
                prediction = predict(model, condition, args.batch_size, device)
                conditions.append({"combination": combination, "position": position, "requested_fraction": fraction, "status": "ok", "actual_missing_fraction": float(removed / eligible) if eligible else 0.0, **score(condition.labels, prediction, threshold)})
                rows.extend({"id": sample_id, "target": float(target), "raw_intensity": float(value), "combination": combination, "position": position, "requested_fraction": fraction} for sample_id, target, value in zip(condition.ids, condition.labels, prediction))
    if len(conditions) != 63 or len(rows) != 727 * 63:
        raise RuntimeError("EF-LSTM Q2 did not produce the full matrix")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "metrics.json", {"method": "EF-LSTM", "source_run": args.run, "split": "test", "mask_sha256": EXPECTED_MASK_SHA256, "n_conditions": 63, "n_predictions": len(rows), "conditions": conditions})
    save_json(output / "manifest.json", {"method": "EF-LSTM", "source_run": args.run, "mask_application": "explicit_text_audio_visual_masks_with_masked_lstm_state", "mask_sha256": EXPECTED_MASK_SHA256, "checkpoint_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
