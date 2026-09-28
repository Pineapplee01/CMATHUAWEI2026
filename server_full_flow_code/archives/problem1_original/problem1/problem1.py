"""问题1任务编排：特征提取、映射与边界验收。"""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from shared.common import config, path
from problem1.pipeline import extract
from problem1.validation import boundary_validation
from problem3.mapping import map_evidence



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['extract','map-evidence','boundary-check'])
    parser.add_argument('--config', default='configs/default.json')
    parser.add_argument('--device')
    parser.add_argument('--annotations', help='人工边界参考 CSV')
    parser.add_argument('--limit', type=int, help='小样本联调，输出独立smoke目录，不作为100条正式结果')
    args = parser.parse_args(); cfg = config(args.config)
    if args.device: cfg['device'] = args.device
    if args.action == 'extract':
        if args.limit is not None:
            if args.limit < 1: parser.error('--limit 必须为正数')
            cfg['q1']['smoke_limit'] = args.limit
            cfg['q1']['output'] = f'problem1/results/smoke_{args.limit}'
            cfg['q1']['work'] = f'problem1/work/smoke_{args.limit}'
        extract(cfg)
    elif args.action == 'map-evidence': map_evidence(cfg)
    else:
        if not args.annotations: parser.error('boundary-check 需要 --annotations')
        boundary_validation(path(args.annotations), path('problem1/results/boundary_validation.json'))


if __name__ == "__main__":
    main()
