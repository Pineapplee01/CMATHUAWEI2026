"""生成问题二专项材料包；明确外部资源、检查50MB与已知本机身份路径。"""
import hashlib
import json
import shutil
import zipfile
from shared.common import ROOT, path, config, json_write
from problem2.verify_submission import verify


def build():
    cfg=config('configs/problem2_robust.json')
    verification=verify(cfg)
    destination=path('deliverables');destination.mkdir(exist_ok=True)
    shutil.copyfile(path(cfg['output'])/'附件3_预测结果.csv',destination/'附件3_预测结果.csv')
    modules=('__init__ adapters averaging completion heads inference losses model problem2 runtime '
             'sentiment_encoder text_encoder temporal training training_step validation evaluation '
             'local_missingness local_evaluation local_report output_contract verify_submission verify_results '
             'package_submission').split()
    files={ROOT/'problem2'/f'{name}.py' for name in modules}
    files.update((ROOT/'shared').glob('*.py'))
    files.update(ROOT/name for name in ['requirements.txt','environment.yml',
                 'problem2/局部缺失建模与题意核对.md','configs/problem2_robust.json',
                 'AAAmodel/roberta-sentiment/provenance.json'])
    files.add(path(cfg['checkpoint']))
    for name in ('robust_main','robust_no_completion','robust_no_reliability'):
        files.add(ROOT/'configs'/f'{name}.json')
        files.add(ROOT/'AAAmodel/checkpoints'/f'{name}.pt')
        for item in ('training_history.csv','run_config.json'):
            files.add(ROOT/'problem2/results'/name/item)
    for folder in (cfg['output'],'problem2/results/local_robustness',
                   'problem2/figures/local_robustness','problem2/figures/robust_selected'):
        files.update(f for f in path(folder).rglob('*') if f.is_file())
    files.update(ROOT/'tests'/name for name in ['__init__.py','test_local_missingness.py',
                                              'test_output_contract.py','test_model_contracts.py'])
    hashes={}
    forbidden=[str(ROOT).encode(), ROOT.parent.name.encode()]
    archive=destination/'问题二_局部缺失模型与预测.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(files):
            content=file.read_bytes()
            if any(token in content for token in forbidden):
                raise ValueError(f'材料包含本机身份路径，未发布: {file.relative_to(ROOT)}')
            name=str(file.relative_to(ROOT));hashes[name]=hashlib.sha256(content).hexdigest()
            z.writestr(name,content)
        z.writestr('SHA256.json',json.dumps(hashes,ensure_ascii=False,indent=2))
        z.writestr('README_专项包.txt',
            '本包仅覆盖问题二，不是三问完整竞赛提交。\n'
            '正式CSV：problem2/results/robust_selected/附件3_预测结果.csv。\n'
            '说明：problem2/局部缺失建模与题意核对.md。\n'
            '运行：python -m problem2.problem2 predict --config configs/problem2_robust.json\n'
            '验证：python -m problem2.verify_submission\n'
            '不是离线自包含包：另需原附件2/3对齐数据、本地原版BERT词表及RoBERTa情感预训练权重，'
            '均放在配置指定的AAAdata/AAAmodel路径，资源文件哈希须与检查点一致。\n'
            '本包包括本题学习的任务参数及消融参数，不包括大体积预训练骨干。\n'
            '已检查已知本机用户名/工作路径；不将该自动检查等同于所有身份信息的人工终审。\n')
    if archive.stat().st_size > 50_000_000:
        raise ValueError('专项包超过50MB')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(name)).hexdigest()==digest for name,digest in hashes.items())
    report={'archive':str(archive.relative_to(ROOT)),'files':len(files),
            'bytes':archive.stat().st_size,'under_50_decimal_MB':True,
            'known_local_identity_scan_passed':True,'sha256_verified':True,
            'self_contained':False,'scope':'problem2 only','verification':verification}
    json_write(destination/'问题二_打包核验.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__': build()
