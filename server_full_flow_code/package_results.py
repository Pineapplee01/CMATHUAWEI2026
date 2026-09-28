"""打包代码、最终任务参数和实际结果；外部骨干资源以哈希清单复现。"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
from shared.common import ROOT, config, path, fingerprint, json_write


def build(cfg):
    output = ROOT / 'deliverables'
    output.mkdir(exist_ok=True)
    files = set()
    for directory in ('problem1', 'problem2', 'problem3', 'shared', 'tests'):
        files.update((ROOT / directory).glob('*.py'))
        files.update((ROOT / directory).glob('*.md'))
        files.update((ROOT / directory).glob('*.m'))
        files.update((ROOT / directory).glob('*.json'))
    for name in ('README.md', '模型完整性检查.md', '运行与优化报告.md',
                 'requirements.txt', 'requirements-lock.txt', 'environment.yml',
                 'environment-lock.yml', 'audit_resources.py', 'package_results.py'):
        files.add(ROOT / name)
    files.update((ROOT / 'configs').glob('*.json'))
    files.add(path(cfg['checkpoint']))
    for resource in ('AAAmodel/roberta-sentiment/provenance.json', 'AAAmodel/bert-base-uncased-sst2/provenance.json'):
        if path(resource).exists(): files.add(path(resource))
    if cfg.get('init_checkpoint'): files.add(path(cfg['init_checkpoint']))
    for directory in ('AAAdata/processed/problem1_covarep', 'AAAdata/processed/problem1_covarep_smoke_1', 'problem1/figures',
                      cfg['output'], cfg['explanation_output'], 'problem3/results/mapping',
                      'problem2/figures/main', 'problem3/figures/main',
                      'problem2/results/optimization', 'problem2/figures/optimization'):
        files.update(f for f in path(directory).rglob('*') if f.is_file())
    for directory in (ROOT / 'problem2/results').glob('opt5_*'):
        files.update(f for f in directory.rglob('*') if f.is_file())
    files.update(f for f in (ROOT / 'problem2/results/five_rounds').rglob('*') if f.is_file())
    files.update(f for f in (ROOT / 'archives/attention_trial').rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    files.update(f for f in (ROOT / 'archives/global_fusion_trial').rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    for directory in (ROOT / 'problem2/results').glob('ote_*'):
        files.update(f for f in directory.rglob('*') if f.is_file())
    files.update(f for f in (ROOT / 'problem2/figures/ote_trials').rglob('*') if f.is_file())
    files.update(f for f in (ROOT / 'problem2/figures/ote_domain_finetune').rglob('*') if f.is_file())
    # 同时保留全部试验训练历史，便于核对优化而非只展示最好一次。
    files.update((ROOT / 'problem2/results').glob('*/training_history.csv'))
    files.update((ROOT / 'problem2/results').glob('*/run_config.json'))
    files.update((ROOT / 'logs').glob('*.log'))
    files.update((ROOT / 'logs').glob('*.json'))
    files = sorted(f for f in files if f.is_file() and f != ROOT / 'logs/package.log')
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    resource = {'encoder_sha256': fingerprint(cfg['bert_path']),
                'external_resources_not_bundled': ['BERT input tokenizer / Q1 weights and selected sentiment encoder weights', 'MFA dictionary/acoustic model',
                    'OpenFace executable/models', 'COVAREP and MATLAB runtime/license', 'original Appendices 1–4'],
                'self_contained': False,
                'note': '包内为代码、特征、任务参数及结果；运行还需上述资源，非离线自包含提交包。'}
    archive = output / 'CPMCM_code_results.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in files: z.write(file, file.relative_to(ROOT))
        z.writestr('SHA256.json', json.dumps(hashes, ensure_ascii=False, indent=2))
        z.writestr('EXTERNAL_RESOURCES.json', json.dumps(resource, ensure_ascii=False, indent=2))
    report = {'archive': str(archive.relative_to(ROOT)), 'files': len(files),
              'bytes': archive.stat().st_size, 'under_50_decimal_MB': archive.stat().st_size <= 50_000_000,
              'self_contained': False, 'external_resources': resource}
    json_write(output / 'package_manifest.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='configs/default.json')
    build(config(parser.parse_args().config))
