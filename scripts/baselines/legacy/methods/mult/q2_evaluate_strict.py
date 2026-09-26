"""Run Q2 for the standalone MulT raw and aligned-control views."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from data_views import resolve_view, run_root
from masked_mult import MaskedMULT
from runtime import COMBINATIONS, EXPECTED_ALIGNED_MASK_SHA256, FRACTIONS, POSITIONS, ROOT, aligned_index, load_bundle, predict, q2_condition, raw_q2_manifest, save_json, score


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--data-view",choices=("raw_unaligned","aligned_50_control"),required=True); parser.add_argument("--run",required=True); parser.add_argument("--device",default="cuda:0"); parser.add_argument("--batch-size",type=int); parser.add_argument("--output-name",default="q2_strict_test_maskaware_v1"); args=parser.parse_args()
    if Path(args.run).name != args.run or Path(args.output_name).name != args.output_name: raise ValueError("run and output names must be directory names")
    view=resolve_view(args.data_view); run=ROOT/run_root(view.name)/args.run; output=run/args.output_name
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    run_manifest=json.loads((run/"q2_mask_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("input_view") != view.name: raise RuntimeError("run manifest belongs to another MulT data view")
    if view.canonical and run_manifest.get("mask_sha256") != EXPECTED_ALIGNED_MASK_SHA256: raise RuntimeError("aligned MulT run is not bound to the common Q2 index")
    config=json.loads((run/"resolved_config.json").read_text(encoding="utf-8")); threshold=json.loads((run/"neutral_interval.json").read_text(encoding="utf-8")); device=torch.device(args.device)
    model=MaskedMULT(**{key:config[key] for key in ("text_dim","audio_dim","vision_dim","model_dim","heads","layers","dropout")}); model.load_state_dict(torch.load(run/"best.pt",map_location=device,weights_only=True)); model.to(device).eval()
    bundle=load_bundle(view.name,"test"); index=aligned_index() if view.canonical else None; batch_size=args.batch_size or (4 if view.name=="raw_unaligned" else 16); output.mkdir(); conditions=[]; rows=[]
    for combination in COMBINATIONS:
        for position in POSITIONS:
            for fraction in FRACTIONS:
                condition,removed,eligible=q2_condition(bundle,view.name,combination,position,fraction,index); prediction=predict(model,condition,batch_size,device)
                conditions.append({"combination":combination,"position":position,"requested_fraction":fraction,"status":"ok","actual_missing_fraction":float(removed/eligible) if eligible else 0.0,**score(condition.labels,prediction,threshold)})
                rows.extend({"id":sample_id,"target":float(target),"raw_intensity":float(value),"combination":combination,"position":position,"requested_fraction":fraction} for sample_id,target,value in zip(condition.ids,condition.labels,prediction))
    if len(conditions)!=63 or len(rows)!=727*63: raise RuntimeError("MulT Q2 did not produce all conditions")
    with (output/"predictions.csv").open("w",newline="",encoding="utf-8") as stream: writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    raw_hash=None
    if not view.canonical:
        payload=raw_q2_manifest(bundle); serialized=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode(); raw_hash=hashlib.sha256(serialized).hexdigest(); save_json(output/"raw_unaligned_q2_manifest.json",{**payload,"mask_sha256":raw_hash})
    save_json(output/"metrics.json",{"method":"MulT","input_view":view.name,"feature_version":view.feature_version,"source_run":args.run,"split":"test","mask_sha256":EXPECTED_ALIGNED_MASK_SHA256 if view.canonical else raw_hash,"n_conditions":63,"n_predictions":len(rows),"conditions":conditions})
    save_json(output/"manifest.json",{"method":"MulT","input_view":view.name,"mask_application":"explicit_text_audio_visual_attention_masks","mask_sha256":EXPECTED_ALIGNED_MASK_SHA256 if view.canonical else raw_hash,"checkpoint_sha256":hashlib.sha256((run/"best.pt").read_bytes()).hexdigest()})


if __name__=="__main__": main()
