"""静态资源盘点工具：不导入模型、不加载权重、不调用外部程序。"""
import argparse
import ast
import importlib.metadata
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.json')
    args = parser.parse_args()
    filename = Path(args.config)
    cfg = json.loads((filename if filename.is_absolute() else ROOT/filename).read_text())
    files = []
    for name in ('config.json','model.safetensors','vocab.txt','tokenizer.json','tokenizer_config.json'):
        files.append(('BERT '+name, ROOT/cfg['bert_path']/name))
    files += [('MFA dictionary', ROOT/cfg['q1']['dictionary']),
              ('MFA acoustic', ROOT/cfg['q1']['acoustic']),
              ('trained checkpoint (expected only after training)', ROOT/cfg['checkpoint'])]
    result = {'files': {name:{'exists':f.is_file(), 'bytes':f.stat().st_size if f.is_file() else None}
                        for name,f in files},
              'executables':{name:shutil.which(name) for name in
                  ('ffmpeg','ffprobe',cfg['q1']['mfa_executable'],cfg['q1']['openface_executable'])},
              'packages':{}, 'syntax':{}}
    for name in ('torch','transformers','opensmile','soundfile','praatio','numpy','pandas','openpyxl','scikit-learn'):
        try: result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['packages'][name] = None
    sources = [Path(__file__)]
    for folder in ('problem1', 'problem2', 'problem3', 'shared', 'tests'):
        sources.extend((ROOT / folder).glob('*.py'))
    for file in sources:
        try:
            ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
            result['syntax'][str(file.relative_to(ROOT))] = 'parsed_without_execution'
        except SyntaxError as exc:
            result['syntax'][str(file.relative_to(ROOT))] = str(exc)
    result['data'] = {'appendix3_aligned':len(list((ROOT/cfg['appendix3']).glob('*.pkl'))),
                      'appendix4_aligned':len(list((ROOT/cfg['appendix4']).glob('*.pkl'))),
                      'split_files':{s:(ROOT/cfg['data_dir']/f'{s}.pkl').is_file() for s in ('train','valid','test')}}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
