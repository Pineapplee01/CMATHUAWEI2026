"""Train standalone MulT on raw_unaligned or aligned_50_control data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from data_views import resolve_view, run_root
from masked_mult import MaskedMULT
from runtime import ROOT, batches, bind_run, calibrate, load_bundle, predict, save_json, score, seed_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-view", choices=("raw_unaligned", "aligned_50_control"), required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if Path(args.run).name != args.run: raise ValueError("--run must be one directory name")
    view = resolve_view(args.data_view)
    run = (ROOT / run_root(view.name) / args.run).resolve()
    if not run.is_relative_to((ROOT / "runs").resolve()) or run.exists(): raise FileExistsError(f"refusing to overwrite {run}")
    run.mkdir(parents=True)
    batch_size = args.batch_size or (2 if view.name == "raw_unaligned" else 16)
    state = {"method": "MulT", "input_view": view.name, "run": args.run, "status": "initializing", "seed": args.seed, "started": time.time()}
    def status(**changes): state.update(changes, updated=time.time()); save_json(run / "status.json", state)
    try:
        seed_all(args.seed); device = torch.device(args.device)
        train, valid, test = (load_bundle(view.name, split) for split in ("train", "valid", "test"))
        if args.smoke:
            train = type(train)(train.ids[:4], {key:value[:4] for key,value in train.values.items()}, {key:value[:4] for key,value in train.native.items()}, train.labels[:4])
            valid = type(valid)(valid.ids[:4], {key:value[:4] for key,value in valid.values.items()}, {key:value[:4] for key,value in valid.native.items()}, valid.labels[:4])
        config = {"text_dim":768,"audio_dim":74,"vision_dim":35,"model_dim":30,"heads":6,"layers":4,"dropout":0.4}
        save_json(run / "resolved_config.json", {**config, "input_view":view.name, "data_root":str(view.data_root), "epochs":args.epochs, "batch_size":batch_size, "learning_rate":args.learning_rate})
        model = MaskedMULT(**config).to(device); optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
        best=float("inf"); history=[]
        for epoch in range(1, (1 if args.smoke else args.epochs)+1):
            status(status="training", epoch=epoch); model.train(); losses=[]
            for batch in batches(train,batch_size,True,args.seed+epoch):
                values={name:torch.as_tensor(batch[name],dtype=torch.float32,device=device) for name in ("text","audio","vision")}
                masks={name:torch.as_tensor(batch["masks"][name],dtype=torch.bool,device=device) for name in ("text","audio","vision")}
                target=torch.as_tensor(batch["labels"],dtype=torch.float32,device=device)
                optimizer.zero_grad(set_to_none=True); loss=torch.nn.functional.mse_loss(model(**values,masks=masks).reshape(-1),target)
                if not torch.isfinite(loss): raise FloatingPointError("non-finite MulT loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),0.6,error_if_nonfinite=True); optimizer.step(); losses.append(float(loss.detach().cpu()))
            prediction=predict(model,valid,batch_size,device); valid_mae=float(np.abs(valid.labels-prediction).mean()); history.append({"epoch":epoch,"train_mse":float(np.mean(losses)),"valid_mae":valid_mae}); save_json(run/"history.json",{"history":history})
            if valid_mae<best: best=valid_mae; torch.save(model.state_dict(),run/"best.pt")
        model.load_state_dict(torch.load(run/"best.pt",map_location=device,weights_only=True)); valid_prediction=predict(model,valid,batch_size,device); threshold=calibrate(valid_prediction,valid.labels); save_json(run/"neutral_interval.json",threshold)
        if args.smoke:
            bind_run(run,view.name)
            status(status="smoke_passed",best_valid_mae=best)
            return
        test_prediction=predict(model,test,batch_size,device); save_json(run/"metrics.json",{"valid":score(valid.labels,valid_prediction,threshold),"test":score(test.labels,test_prediction,threshold)})
        bind_run(run,view.name); save_json(run/"protocol_manifest.json",{"method":"MulT","input_view":view.name,"feature_version":view.feature_version,"data_root":str(view.data_root),"checkpoint":str(run/"best.pt"),"checkpoint_sha256":hashlib.sha256((run/"best.pt").read_bytes()).hexdigest(),"threshold_source":str(run/"neutral_interval.json"),"selection_split":"valid"}); status(status="completed",best_valid_mae=best,test_evaluated=True)
    except Exception as error:
        status(status="failed",error=repr(error),traceback=traceback.format_exc()); raise


if __name__ == "__main__": main()
