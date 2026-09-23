"""Run from the project root; never consumes data outside data/."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

from aumdf.data import PROJECT_ROOT
from aumdf.engine import artifact_path, evaluate_checkpoint, train_run, write_json

def main():
    parser=argparse.ArgumentParser(description="Independent AUMDF competition-data reproduction")
    commands=parser.add_subparsers(dest="command",required=True)
    train=commands.add_parser("train")
    train.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/adapted.yaml")
    train.add_argument("--output",type=Path)
    train.add_argument("--device",default="auto")
    train.add_argument("--smoke",action="store_true",help="1 epoch/stage, 64 train and 32 valid; not a result reproduction")
    train.add_argument("--missing-mode",choices=("random","block"))
    evaluate=commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint",required=True,type=Path)
    evaluate.add_argument("--split",choices=("train","valid","test"),default="valid")
    evaluate.add_argument("--output",required=True,type=Path)
    evaluate.add_argument("--device",default="auto")
    evaluate.add_argument("--whole-modalities",action="store_true")
    args=parser.parse_args()
    if args.command=="train":
        config=yaml.safe_load(args.config.read_text(encoding="utf-8"))
        config=copy.deepcopy(config)
        if args.smoke:
            config["training"].update(teacher_epochs=1,student_epochs=1)
        if args.missing_mode:
            config["missingness"]["mode"]=args.missing_mode
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output=args.output or PROJECT_ROOT/"artifacts/aumdf"/(("smoke-" if args.smoke else "run-")+stamp)
        result=train_run(config,output,args.device,
                         train_limit=64 if args.smoke else None,valid_limit=32 if args.smoke else None)
    else:
        output=artifact_path(PROJECT_ROOT,args.output)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        result=evaluate_checkpoint(args.checkpoint,args.split,args.device,whole_modalities=args.whole_modalities)
        output.parent.mkdir(parents=True,exist_ok=True)
        write_json(output,result)
        result={**result,"conditions":[{k:v for k,v in c.items() if k!="predictions"} for c in result["conditions"]]}
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=="__main__":
    main()
