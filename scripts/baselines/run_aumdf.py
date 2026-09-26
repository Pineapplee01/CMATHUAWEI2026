"""Run from the project root; never consumes data outside data/."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml

# Support the documented standalone command without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from e_emotion.evaluation import write_predictions_csv

from e_emotion.baselines.aumdf.data import PROJECT_ROOT
from e_emotion.baselines.aumdf.engine import artifact_path, evaluate_checkpoint, train_run, write_json

def main():
    parser=argparse.ArgumentParser(description="Independent AUMDF competition-data reproduction")
    commands=parser.add_subparsers(dest="command",required=True)
    train=commands.add_parser("train")
    train.add_argument("--config",type=Path,default=PROJECT_ROOT/"configs/aumdf/adapted.yaml")
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
                         train_limit=64 if args.smoke else None,valid_limit=32 if args.smoke else None,
                         strict_data=True)
    else:
        output=artifact_path(PROJECT_ROOT,args.output)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        csv_dir = output.with_suffix("")
        if csv_dir == output or csv_dir.exists():
            raise FileExistsError("use a new .json output name with a new sibling CSV directory")
        result=evaluate_checkpoint(args.checkpoint,args.split,args.device,whole_modalities=args.whole_modalities,
                                   strict_data=True)
        output.parent.mkdir(parents=True,exist_ok=True)
        write_json(output,result)
        for condition in result["conditions"]:
            write_predictions_csv(csv_dir / (condition["name"] + ".csv"), condition["predictions"])
        result={**result,"conditions":[{k:v for k,v in c.items() if k!="predictions"} for c in result["conditions"]]}
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=="__main__":
    main()
