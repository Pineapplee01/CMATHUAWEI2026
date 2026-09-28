"""问题2训练、评估和专项预测的命令编排。"""
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from shared.common import config
from problem2.training import train
from problem2.validation import evaluate_suite
from problem2.inference import predict_appendix3
from problem2.runtime import restore



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'evaluate', 'predict'])
    parser.add_argument('--config', default='configs/problem2_robust.json')
    parser.add_argument('--checkpoint')
    parser.add_argument('--device')
    parser.add_argument('--split', choices=['valid', 'test'], default='valid')
    parser.add_argument('--frozen-test', action='store_true', help='声明配置已冻结才评估 test')
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--quick', action='store_true', help='只评估完整与30%三模态缺失，不作全场景扫描')
    args = parser.parse_args(); cfg = config(args.config)
    if args.device: cfg['device'] = args.device
    if args.action == 'train':
        train(cfg)
    else:
        if args.split == 'test' and not args.frozen_test:
            parser.error('test 只在方案冻结后使用，请显式传 --frozen-test')
        model, saved = restore(args.checkpoint or cfg['checkpoint'], cfg['device'])
        # 运行路径可以改变，模型/掩码/文本接口必须来自检查点。
        for key in ('output', 'data_dir', 'appendix3', 'appendix4'):
            saved[key] = cfg[key]
        saved['checkpoint'] = args.checkpoint or cfg['checkpoint']
        saved['output_rule'] = cfg.get('output_rule', 'independent_heads')
        if args.action == 'evaluate': evaluate_suite(model, saved, args.split, args.noise, args.quick)
        else: predict_appendix3(model, saved)


if __name__ == "__main__":
    main()
