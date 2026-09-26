"""Two-stage training, validation-only checkpoint choice, and frozen evaluation."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from e_emotion.evaluation import adapt_output, prediction_rows, score_records
from e_emotion.evaluation.outputs import PROTOCOL_VERSION

from e_emotion.baselines.aumdf.data import CANONICAL_PROCESSED_ROOT, PROJECT_ROOT, load_attachment2, resolve_source_path
from e_emotion.baselines.aumdf.losses import DistillationLoss, freeze_teacher
from e_emotion.baselines.aumdf.metrics import metrics, select_neutral_threshold
from e_emotion.baselines.aumdf.missingness import corrupt
from e_emotion.baselines.aumdf.model import AUMDF, MODALITIES, ModelConfig


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(path):
    """Hash each immutable split when the source is a processed directory."""
    source = Path(path)
    if source.is_file():
        return {source.name: sha256(source)}
    return {item.name: sha256(item) for item in sorted(source.glob("*.npz"))}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(value)


def artifact_path(root, path):
    candidate = Path(path)
    candidate = (Path(root) / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not candidate.is_relative_to((Path(root) / "artifacts").resolve()):
        raise ValueError("outputs/checkpoints must be inside project artifacts/")
    return candidate


def transfer(batch, device, rate=0.0, mode="random", generator=None, modalities=MODALITIES):
    features, simulated = corrupt(batch["features"], batch["valid"], rate, mode=mode,
                                  generator=generator, modalities=modalities)
    valid = {m: batch["valid"][m].to(device) for m in MODALITIES}
    observed = {m: (simulated[m] & batch["observed"][m]).to(device) for m in MODALITIES}
    return {m: features[m].to(device) for m in MODALITIES}, valid, observed, batch["target"].to(device)


@torch.no_grad()
def predict(model, loader, device, rate=0.0, mode="random", seed=2026, modalities=MODALITIES):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    predictions, targets, ids = [], [], []
    for batch in loader:
        x, valid, observed, y = transfer(batch, device, rate, mode, generator, modalities)
        output = model(x, valid, observed)
        predictions.extend(output.score.cpu().tolist())
        targets.extend(y.cpu().tolist())
        ids.extend(batch["id"])
    adapted = adapt_output(predictions)
    return np.asarray(targets), adapted.intensity, ids, adapted.raw_intensity


def save_checkpoint(path, model, criterion, config, scaler, stage, epoch, validation_score,
                    threshold=0.0, run_metadata=None):
    # Portable trusted local tensors; no optimizer state or executable Python objects.
    payload = {"model": model.state_dict(), "loss": criterion.state_dict(),
               "model_config": asdict(model.config), "config": config, "scaler": scaler,
               "stage": stage, "epoch": epoch, "validation_score": validation_score,
               "neutral_threshold": threshold, "implementation": "independent_competition_adaptation"}
    payload["scoring_protocol_version"] = PROTOCOL_VERSION
    payload["training_provenance"] = dict(run_metadata or {})
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = ModelConfig(**checkpoint["model_config"])
    model = AUMDF(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def training_provenance(checkpoint, path):
    """Preserve smoke status, including checkpoints from the first local run."""
    provenance = checkpoint.get("training_provenance")
    if provenance:
        return provenance
    # Legacy locally generated checkpoints have adjacent, recorded run summaries.
    summary_path = Path(path).parent / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {key: summary.get(key) for key in ("scope", "used_train", "used_valid")}
    return {"scope": "unknown", "used_train": None, "used_valid": None}


def train_run(config, output_dir, device="auto", project_root=PROJECT_ROOT,
              train_limit=None, valid_limit=None, *, strict_data=True,
              canonical_root=CANONICAL_PROCESSED_ROOT, fair_datasets=None):
    output = Path(output_dir).resolve() if fair_datasets is not None else artifact_path(project_root, output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {output}")
    strict_data = bool(strict_data)
    training = config["training"]
    seed = int(training.get("seed",2026))
    for name in ("teacher_epochs", "student_epochs", "batch_size", "patience"):
        if int(training.get(name, 20 if name != "batch_size" else 32)) < 1:
            raise ValueError(f"{name} must be positive")
    seed_everything(seed)
    torch.set_num_threads(2)
    device = choose_device(device)
    model_config = ModelConfig(**config.get("model",{}))
    if fair_datasets is None:
        datasets, scaler, audit = load_attachment2(
            config["data_file"], project_root=project_root, expected_dims=model_config.input_dims,
            strict=strict_data, canonical_root=canonical_root,
        )
    else:
        datasets = fair_datasets
        if not all(name in datasets and len(datasets[name]) for name in ("train", "valid")):
            raise ValueError("fair_datasets must contain nonempty train and valid datasets")
        scaler = {name: {"mean": [0.0] * width, "std": [1.0] * width}
                  for name, width in zip(MODALITIES, model_config.input_dims)}
        audit = {"data_format": "problem2-fair-v1", "view": config.get("fair_view"),
                 "normalization_source": "processed_po; identity adapter"}
    selected = {}
    for split, limit in (("train",train_limit),("valid",valid_limit)):
        if limit is not None and limit < 2:
            raise ValueError("smoke limits must be at least 2")
        selected[split] = Subset(datasets[split], range(min(limit,len(datasets[split])))) if limit else datasets[split]
    batch_size = int(training.get("batch_size",32))
    shuffle_generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(selected["train"], batch_size=batch_size, shuffle=True,
                            generator=shuffle_generator, num_workers=0),
        "valid": DataLoader(selected["valid"], batch_size=batch_size, shuffle=False, num_workers=0),
    }
    output.mkdir(parents=True, exist_ok=False)
    scope = "smoke_only" if train_limit or valid_limit else (
        "problem2_fair_v1" if fair_datasets is not None else "competition_subset_reproduction"
    )
    run_metadata = {"scope":scope, "used_train":len(selected["train"]), "used_valid":len(selected["valid"])}
    environment = {"python":platform.python_version(), "torch":str(torch.__version__),
                   "numpy":np.__version__, "cuda":torch.version.cuda, "device":str(device),
                   "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None}
    write_json(output / "config.json", config)
    if fair_datasets is None:
        source = resolve_source_path(project_root, config["data_file"], strict=strict_data,
                                     canonical_root=canonical_root)
        data_sha256 = source_sha256(source)
    else:
        source = config.get("fair_data_root", "injected_problem2_fair")
        data_sha256 = None
    write_json(output / "data_audit.json", {**audit,"used_train":len(selected["train"]),
                                      "used_valid":len(selected["valid"]),
                                      "data_source":str(source),
                                      "data_sha256":data_sha256})
    write_json(output / "environment.json", environment)
    write_json(output / "status.json", {"state":"running","scope":scope})
    missing = config.get("missingness",{})
    rates = list(missing.get("rates",[.1,.2,.3,.4,.5,.6,.7]))
    if not rates or any(not 0 <= float(r) <= 1 for r in rates):
        raise ValueError("training missing rates must be nonempty and inside [0,1]")
    mode = missing.get("mode","random")
    validation_rate = float(missing.get("validation_rate",.3))
    teacher, student = AUMDF(model_config).to(device), AUMDF(model_config).to(device)
    summary = {"scope":scope, "environment":environment,
               "parameters_per_network":sum(p.numel() for p in teacher.parameters()),
               "used_train":len(selected["train"]), "used_valid":len(selected["valid"]),
               "test_used_for_selection":False, "stages":[]}
    start = time.monotonic()
    try:
        for stage, model in (("teacher",teacher),("student",student)):
            stage_start = time.monotonic()
            criterion = DistillationLoss(hidden_dim=model_config.hidden_dim, **config.get("loss",{})).to(device)
            if stage == "student":
                teacher, _ = restore(output / "teacher.pt", device)
                freeze_teacher(teacher)
            parameters = list(model.parameters()) + (list(criterion.parameters()) if stage == "student" else [])
            optimizer = torch.optim.Adam(parameters, lr=float(training.get("learning_rate",.001)))
            augmentation_rng = torch.Generator().manual_seed(seed + (100 if stage=="student" else 0))
            best, best_epoch, completed = float("inf"), 0, 0
            for epoch in range(1, int(training.get(stage+"_epochs",20))+1):
                tick = time.monotonic()
                model.train()
                sums, count = {}, 0
                for batch in loaders["train"]:
                    index = int(torch.randint(len(rates),(1,),generator=augmentation_rng))
                    rate = float(rates[index]) if stage=="student" else 0.0
                    x, valid, observed, targets = transfer(batch, device, rate, mode, augmentation_rng)
                    optimizer.zero_grad(set_to_none=True)
                    output_student = model(x, valid, observed)
                    if stage == "student":
                        with torch.no_grad():
                            teacher_x = {m:batch["features"][m].to(device) for m in MODALITIES}
                            teacher_mask = {m:batch["observed"][m].to(device) for m in MODALITIES}
                            target_output = teacher(teacher_x, valid, teacher_mask)
                        losses = criterion(model, output_student, target_output, targets)
                    else:
                        task = criterion.task(output_student,targets)
                        regularization = criterion.qkv_coefficient * model.qkv_penalty()
                        losses = {"total":task+criterion.lambda_reg*regularization,
                                  "task":task,"regularization":regularization}
                    if not all(torch.isfinite(v).all() for v in losses.values()):
                        raise FloatingPointError(f"nonfinite loss in {stage} epoch {epoch}")
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(parameters,float(training.get("gradient_clip",1.0)),
                                                  error_if_nonfinite=True)
                    optimizer.step()
                    n = len(targets)
                    count += n
                    for key,value in losses.items():
                        sums[key] = sums.get(key,0.0) + float(value.detach())*n
                y,p,_,_ = predict(model,loaders["valid"],device,seed=seed+1000)
                clean = metrics(y,p)
                damaged = None
                if stage=="student":
                    yy,pp,_,_ = predict(model,loaders["valid"],device,validation_rate,mode,seed+1000)
                    damaged = metrics(yy,pp)
                selection = (clean["mae"]+damaged["mae"])/2 if damaged else clean["mae"]
                if selection < best:
                    best,best_epoch = selection,epoch
                    save_checkpoint(output/(stage+".pt"),model,criterion,config,scaler,stage,epoch,best,
                                    run_metadata=run_metadata)
                completed = epoch
                record = {"stage":stage,"epoch":epoch,"train_loss":{k:v/count for k,v in sums.items()},
                          "valid_clean":clean,"valid_missing":damaged,"selection_mae":selection,
                          "best_epoch":best_epoch,"seconds":time.monotonic()-tick}
                with (output/"history.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
                print(json.dumps({"stage":stage,"epoch":epoch,"loss":round(sums["total"]/count,4),
                                  "valid_mae":round(clean["mae"],4),"selection_mae":round(selection,4),
                                  "best_epoch":best_epoch,"seconds":round(record["seconds"],2)}),flush=True)
                if epoch-best_epoch >= int(training.get("patience",20)):
                    break
            best_model,checkpoint = restore(output/(stage+".pt"),device)
            y,p,_,_ = predict(best_model,loaders["valid"],device,seed=seed+1000)
            threshold = select_neutral_threshold(y,p,split="valid")
            checkpoint["neutral_threshold"] = threshold
            torch.save(checkpoint,output/(stage+".pt"))
            summary[stage] = {"epochs_completed":completed,"best_epoch":best_epoch,
                              "selection_mae":best,"valid_clean":metrics(y,p,threshold),
                              "seconds":time.monotonic()-stage_start}
            if stage=="student":
                yy,pp,_,_ = predict(best_model,loaders["valid"],device,validation_rate,mode,seed+1000)
                summary[stage]["valid_missing"] = metrics(yy,pp,threshold)
            summary["stages"].append(stage)
        summary["seconds"] = time.monotonic()-start
        if device.type=="cuda":
            summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        write_json(output/"summary.json",summary)
        write_json(output/"status.json",{"state":"complete","scope":scope})
        return summary
    except Exception as exc:
        write_json(output/"status.json",{"state":"failed","scope":scope,"error":str(exc)})
        raise


def evaluate_checkpoint(checkpoint_path, split="valid", device="auto", project_root=PROJECT_ROOT,
                        rates=(0,.1,.3,.5,.7), modes=("random","block"), whole_modalities=False,
                        *, strict_data=True, canonical_root=CANONICAL_PROCESSED_ROOT):
    if split not in {"train","valid","test"}:
        raise ValueError("split must be train, valid or test")
    device = choose_device(device)
    path = artifact_path(project_root,checkpoint_path)
    model,checkpoint = restore(path,device)
    provenance = training_provenance(checkpoint,path)
    config=checkpoint["config"]
    strict_data = bool(strict_data)
    datasets,_,audit=load_attachment2(config["data_file"],project_root=project_root,
                                     expected_dims=model.config.input_dims,scaler=checkpoint["scaler"],
                                     strict=strict_data, canonical_root=canonical_root)
    loader=DataLoader(datasets[split],batch_size=int(config["training"].get("batch_size",32)),shuffle=False)
    threshold=float(checkpoint["neutral_threshold"])
    seed=int(config["training"].get("seed",2026))+1000
    conditions=[]
    plans=[("complete",0.0,"random",MODALITIES)]
    plans += [(f"{mode}_{rate:g}",float(rate),mode,MODALITIES) for mode in modes for rate in rates if rate>0]
    if whole_modalities:
        for size in (1,2):
            for keep in itertools.combinations(MODALITIES,size):
                plans.append(("only_"+"_".join(keep),1.0,"whole",tuple(m for m in MODALITIES if m not in keep)))
    for name,rate,mode,modalities in plans:
        y,p,ids,raw=predict(model,loader,device,rate,mode,seed,modalities)
        adapted = adapt_output(raw, threshold=threshold)
        rows = prediction_rows(ids, adapted)
        truth = [{"id": sample_id, "intensity": float(target)} for sample_id, target in zip(ids,y)]
        scores = metrics(y,p,threshold)
        # Score the very same records that run.py exports, including exact ID coverage.
        scores["competition"] = score_records(truth, rows)
        for row, target in zip(rows,y):
            row["target"] = float(target)
        conditions.append({"name":name,"rate":rate,"mode":mode,"metrics":scores,
                           "output_adapter":adapted.metadata,"predictions":rows})
    evaluation_scope = {"smoke_only":"smoke_evaluation",
                        "competition_subset_reproduction":"competition_subset_evaluation"}.get(
                            provenance["scope"], "unknown_training_scope")
    return {"protocol_version":PROTOCOL_VERSION,
            "checkpoint":str(path),"checkpoint_sha256":sha256(path),"split":split,
            "scope":evaluation_scope,"training_provenance":provenance,
            "neutral_threshold_source":"best checkpoint clean valid",
            "conditions":conditions,"data_audit":audit}
