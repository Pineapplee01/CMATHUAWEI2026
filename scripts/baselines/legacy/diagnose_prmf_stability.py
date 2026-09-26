#!/usr/bin/env python3
"""Deterministically replay P-RMF training and record its first non-finite value.

This diagnostic intentionally leaves the baseline implementation untouched.  It
mirrors the competition adapter's seed, data, optimizer groups, scheduler, and
gradient repair policy, then records VAE log-variance bounds, proxy-weight
finiteness, gradients, parameters, and Adam moments per batch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent
PROCESSED = REFERENCE.parents[1] / "AAAdata" / "Appendix_2" / "标准化" / "对齐版本" / "processed"
DEFAULT_OUTPUT = ROOT / "runs" / "diagnostics" / "maskaware_nan_replay_seed2026_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=39)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def finite_summary(tensor: torch.Tensor) -> dict[str, float | bool]:
    finite = torch.isfinite(tensor)
    summary: dict[str, float | bool] = {
        "finite": bool(finite.all()),
        "finite_fraction": float(finite.float().mean()),
    }
    if bool(finite.any()):
        values = tensor[finite]
        summary.update(min=float(values.min()), max=float(values.max()), absmax=float(values.abs().max()))
    return summary


class TrackedLoader:
    def __init__(self, loader: DataLoader, state: dict[str, int]) -> None:
        self.loader = loader
        self.state = state

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch, value in enumerate(self.loader):
            self.state["batch"] = batch
            yield value


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def first_nonfinite_named(values: list[tuple[str, torch.Tensor]]) -> str | None:
    for name, value in values:
        if not bool(torch.isfinite(value).all()):
            return name
    return None


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    output = args.output.resolve()
    runs = (ROOT / "runs").resolve()
    if not output.is_relative_to(runs):
        raise ValueError("diagnostic output must be inside P-RMF/runs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {output}")

    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(REFERENCE / "protocol_src"))
    # The upstream module parses sys.argv at import time.  Its native launcher
    # uses the same temporary argv isolation before importing this module.
    caller_argv = sys.argv
    sys.argv = ["train.py"]
    try:
        import train as native
    finally:
        sys.argv = caller_argv
    from cpmcm_extra import CanonicalDataset
    from core.losses import MultimodalLoss
    from core.scheduler import get_scheduler
    from models.P_RMF import build_model

    config = yaml.safe_load((ROOT / "cpmcm_train_mosei.yaml").read_text(encoding="utf-8"))
    config["dataset"]["dataPath"] = str(PROCESSED)
    config["model"]["feature_extractor"]["input_length"] = [50, 50, 50]
    config["base"]["seed"] = args.seed
    model = build_model(config).cuda()
    groups = [
        {"params": getattr(model, attribute).parameters(), "lr": learning_rate}
        for attribute, learning_rate in [
            ("bertmodel", 5e-6),
            ("crossmodal_encoder", 5e-6),
            *[(attribute, config["base"]["lr"]) for attribute in (
                "proj_l", "proj_a", "proj_v", "generate_proxy_modality", "fc1", "fc2"
            )],
        ]
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=config["base"]["weight_decay"])
    scheduler = get_scheduler(optimizer, config)
    loss = MultimodalLoss(config)
    metrics = native.MetricsTop(train_mode=config["base"]["train_mode"]).getMetics("mosei")
    state: dict[str, int] = {"epoch": 0, "batch": 0, "optimizer_steps": 0, "gradient_repairs": 0}
    trace: dict = {
        "kind": "P-RMF numerical-stability replay",
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "device": str(torch.cuda.current_device()),
        "processed": str(PROCESSED),
        "batches": [],
        "failure": None,
    }

    latest_vae: dict[str, dict] = {}
    latest_proxy: dict[str, object] = {}

    def logvar_hook(name: str):
        def hook(_module, _inputs, output):
            logvar = output.detach()
            std = torch.exp(0.5 * logvar)
            raw_score = 1.0 / std
            latest_vae[name] = {
                "log_var": finite_summary(logvar),
                "std": finite_summary(std),
                "inverse_std": finite_summary(raw_score),
                "unstable_score_count": int((raw_score > 80).sum().item()),
            }
        return hook

    for name, vae in (("text", model.generate_proxy_modality.text_VAE),
                      ("vision", model.generate_proxy_modality.image_VAE),
                      ("audio", model.generate_proxy_modality.audio_VAE)):
        vae.encoder.fc2_log_var.register_forward_hook(logvar_hook(name))

    def proxy_hook(_module, _inputs, output):
        kl_loss, proxy, weights = output
        latest_proxy.update(
            kl_loss=finite_summary(kl_loss.detach()),
            proxy=finite_summary(proxy.detach()),
            weights=finite_summary(weights.detach()),
            first_nonfinite=first_nonfinite_named([
                ("kl_loss", kl_loss.detach()), ("proxy", proxy.detach()), ("weights", weights.detach()),
            ]),
        )

    model.generate_proxy_modality.register_forward_hook(proxy_hook)
    original_step = optimizer.step

    def traced_step(*step_args, **step_kwargs):
        gradients = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
        bad_gradient = first_nonfinite_named(gradients)
        if bad_gradient is not None:
            state["gradient_repairs"] += 1
            for _, gradient in gradients:
                torch.nan_to_num_(gradient, nan=0.0, posinf=1e3, neginf=-1e3)
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in gradients], max_norm=5.0, error_if_nonfinite=True)
        result = original_step(*step_args, **step_kwargs)
        state["optimizer_steps"] += 1
        bad_parameter = first_nonfinite_named(list(model.named_parameters()))
        moment_values = []
        for parameter, values in optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in values:
                    moment_values.append((f"{key}:{tuple(parameter.shape)}", values[key]))
        bad_moment = first_nonfinite_named(moment_values)
        row = {
            "epoch": state["epoch"],
            "batch": state["batch"],
            "optimizer_step": state["optimizer_steps"],
            "first_nonfinite_gradient": bad_gradient,
            "first_nonfinite_parameter_after_step": bad_parameter,
            "first_nonfinite_adam_moment_after_step": bad_moment,
            "vae": latest_vae,
            "proxy": latest_proxy,
        }
        if bad_gradient is not None or bad_parameter is not None or bad_moment is not None or latest_proxy.get("first_nonfinite"):
            trace["failure"] = row
            save(output, trace)
            raise FloatingPointError("diagnostic captured non-finite training state")
        return result

    optimizer.step = traced_step
    train_loader = DataLoader(
        CanonicalDataset(config["dataset"]["dataPath"], "train"),
        batch_size=config["base"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        CanonicalDataset(config["dataset"]["dataPath"], "valid"),
        batch_size=config["base"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    try:
        for epoch in range(1, args.epochs + 1):
            state["epoch"] = epoch
            native.train(model, TrackedLoader(train_loader, state), optimizer, loss, epoch, metrics)
            scheduler.step()
            model.eval()
            with torch.no_grad():
                for batch_index, batch in enumerate(valid_loader):
                    state["batch"] = batch_index
                    inputs = tuple(batch[key].cuda() for key in ("vision_m", "audio_m", "text_m"))
                    masks = tuple(batch[key].cuda() for key in ("mV", "mA", "mT"))
                    output_value = model((None, None, None), inputs, observed_masks=masks)["sentiment_preds"]
                    if not bool(torch.isfinite(output_value).all()):
                        trace["failure"] = {
                            "epoch": epoch,
                            "batch": batch_index,
                            "stage": "validation_forward",
                            "vae": latest_vae,
                            "proxy": latest_proxy,
                        }
                        save(output, trace)
                        raise FloatingPointError("diagnostic captured non-finite validation prediction")
            trace["batches"].append({
                "epoch": epoch,
                "optimizer_steps": state["optimizer_steps"],
                "gradient_repairs": state["gradient_repairs"],
                "vae": latest_vae,
                "proxy": latest_proxy,
            })
            save(output, trace)
    except Exception as error:
        if trace["failure"] is None:
            trace["failure"] = {"error": repr(error), "epoch": state["epoch"], "batch": state["batch"]}
            save(output, trace)
        raise

    trace["completed"] = True
    trace["finished"] = time.time()
    save(output, trace)


if __name__ == "__main__":
    main()
