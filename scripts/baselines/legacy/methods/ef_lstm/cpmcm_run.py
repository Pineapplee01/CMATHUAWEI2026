"""Train the independent canonical-NPZ EF-LSTM baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from model import MaskAwareEFLSTM
from runtime import CANONICAL_ROOT, ROOT, batches, bind_common_q2, calibrate, load_bundle, predict, save_json, score, seed_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError("--run must be one directory name")
    run = (ROOT / "runs" / args.run).resolve()
    if not run.is_relative_to((ROOT / "runs").resolve()) or run.exists():
        raise FileExistsError(f"refusing to overwrite run {run}")
    run.mkdir(parents=True)
    state = {"method": "EF-LSTM", "run": args.run, "status": "initializing", "seed": args.seed, "started": time.time()}

    def status(**changes):
        state.update(changes, updated=time.time())
        save_json(run / "status.json", state)

    try:
        seed_all(args.seed)
        device = torch.device(args.device)
        train, valid, test = (load_bundle(split) for split in ("train", "valid", "test"))
        if args.smoke:
            train = type(train)(train.ids[:16], {key: value[:16] for key, value in train.values.items()}, {key: value[:16] for key, value in train.native.items()}, train.labels[:16])
            valid = type(valid)(valid.ids[:16], {key: value[:16] for key, value in valid.values.items()}, {key: value[:16] for key, value in valid.native.items()}, valid.labels[:16])
        config = {"text_dim": 768, "audio_dim": 74, "vision_dim": 35, "hidden_dim": 256, "num_layers": 2, "dropout": 0.4}
        save_json(run / "resolved_config.json", {**config, "data_root": str(CANONICAL_ROOT), "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate})
        model = MaskAwareEFLSTM(**config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        best = float("inf")
        history = []
        for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
            status(status="training", epoch=epoch)
            model.train()
            losses = []
            for batch in batches(train, args.batch_size, True, args.seed + epoch):
                inputs = {name: torch.as_tensor(batch[name], dtype=torch.float32, device=device) for name in ("text", "audio", "vision")}
                masks = {name: torch.as_tensor(batch["masks"][name], dtype=torch.bool, device=device) for name in ("text", "audio", "vision")}
                target = torch.as_tensor(batch["labels"], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(**inputs, masks=masks).reshape(-1), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite EF-LSTM loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            valid_prediction = predict(model, valid, args.batch_size, device)
            valid_mae = float(np.abs(valid.labels - valid_prediction).mean())
            history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "valid_mae": valid_mae})
            save_json(run / "history.json", {"history": history})
            if valid_mae < best:
                best = valid_mae
                torch.save(model.state_dict(), run / "best.pt")
        model.load_state_dict(torch.load(run / "best.pt", map_location=device, weights_only=True))
        valid_prediction = predict(model, valid, args.batch_size, device)
        threshold = calibrate(valid_prediction, valid.labels)
        save_json(run / "neutral_interval.json", threshold)
        if args.smoke:
            bind_common_q2(run)
            status(status="smoke_passed", best_valid_mae=best)
            return
        test_prediction = predict(model, test, args.batch_size, device)
        np.savetxt(run / "test_predictions.csv", np.column_stack((test.labels, test_prediction)), delimiter=",", header="target,raw_intensity", comments="")
        save_json(run / "metrics.json", {"valid": score(valid.labels, valid_prediction, threshold), "test": score(test.labels, test_prediction, threshold)})
        bind_common_q2(run)
        save_json(run / "protocol_manifest.json", {"method": "EF-LSTM", "canonical_root": str(CANONICAL_ROOT), "checkpoint": str(run / "best.pt"), "checkpoint_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest(), "threshold_source": str(run / "neutral_interval.json"), "selection_split": "valid", "q2_mask_sha256": "bddfc02985e9528bcab58a252ebb9beaa8c6a9812b609b6b88c7a547fd623916"})
        status(status="completed", best_valid_mae=best, test_evaluated=True)
    except Exception as error:
        status(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
