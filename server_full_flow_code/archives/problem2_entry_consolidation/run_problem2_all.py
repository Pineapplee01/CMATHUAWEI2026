"""问题二总入口：代码检查、正式模型完整评估、附件3预测、六组多seed消融及绘图。"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def read(filename):
    p=Path(filename)
    return json.loads((p if p.is_absolute() else ROOT/p).read_text(encoding='utf-8'))


def write(filename,data):
    filename.parent.mkdir(parents=True,exist_ok=True)
    temp=filename.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(filename)


def digest(filename):
    h=hashlib.sha256()
    with filename.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['full','formal','ablation'],default='full',
                   help='full全部流程；formal只评估正式权重；ablation只运行六组研究实验')
    p.add_argument('--config',default='configs/problem2_ablation.json',help='六组研究实验配置')
    p.add_argument('--seeds',nargs='+',type=int,help='训练seed，省略时读取实验配置中的5个seed')
    p.add_argument('--device',help='例如cuda:3；默认读取实验配置')
    p.add_argument('--run-name',default='problem2_complete',help='独立运行目录名，不覆盖历史结果')
    p.add_argument('--resume',action='store_true',help='跳过已成功阶段，继续失败或未开始的阶段')
    p.add_argument('--smoke',action='store_true',help='仅缩小消融训练/评估；正式模型检查仍使用全量数据')
    p.add_argument('--dry-run',action='store_true',help='只显示完整执行计划，不创建目录或启动训练')
    a=p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_name):p.error('run-name只允许字母、数字、下划线和连字符')
    if a.seeds is not None and (len(set(a.seeds))!=len(a.seeds) or any(not 0<=s<2**32 for s in a.seeds)):
        p.error('seeds必须互不重复且位于0到4294967295')
    if a.smoke and a.mode=='formal':p.error('formal模式无小样本版本，请去掉--smoke')
    return a


def plan(args,root,spec):
    python=sys.executable
    stages=[('01_code_checks',[python,'-m','unittest','discover','-s','tests','-q'])]
    if args.mode in ('full','formal'):
        common=['--config',str(root/'formal_config.json'),'--device',spec['device']]
        stages += [('02_formal_test',[python,'-m','problem2.problem2','evaluate','--split','test','--frozen-test','--scenarios',*common]),
                   ('03_formal_valid_scenarios',[python,'-m','problem2.problem2','evaluate','--split','valid','--scenarios',*common]),
                   ('04_appendix3',[python,'-m','problem2.problem2','predict',*common])]
    if args.mode in ('full','ablation'):
        command=[python,'-m','problem2.ablation','--config',str(root/'ablation_config.json'),
                 '--run-name',args.run_name,'--device',spec['device'],'--seeds',*map(str,spec['seeds'])]
        if args.smoke:command+=['--smoke']
        if args.resume:command+=['--resume']
        stages.append(('05_ablation_and_figures',command))
    return stages


def execute(name,command,root):
    log=root/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
    print(f'\n[{name}] 日志：{log}',flush=True)
    with log.open('a',encoding='utf-8') as stream:
        stream.write(f'\nSTART {datetime.now(timezone.utc).isoformat()}\n')
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace',bufsize=1)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    if code:raise subprocess.CalledProcessError(code,command)


def main(argv=None):
    args=arguments(argv);spec=read(args.config)
    spec['device']=args.device or spec['device'];spec['seeds']=args.seeds or spec['seeds']
    suffix='_smoke' if args.smoke and not args.run_name.endswith('_smoke') else ''
    name=args.run_name+suffix
    root=ROOT/'problem2/results/full_runs'/name
    stages=plan(args,root,spec)
    print(f'模式={args.mode}；设备={spec["device"]}；seeds={spec["seeds"]}；输出={root}',flush=True)
    if args.dry_run:
        import shlex
        for stage,cmd in stages:print(stage+': '+shlex.join(cmd))
        return
    formal=read('configs/problem2_aligned.json')
    checkpoint=Path(formal['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else ROOT/checkpoint
    identity={'mode':args.mode,'smoke':args.smoke,'spec':spec,'formal_config':formal,
        'base_config':read(spec['base_config']),'formal_checkpoint_sha256':digest(checkpoint),
        'data':{str(f):digest(f) for f in [*(Path(formal['data_dir'])/(n+'.npz') for n in ('train','valid','test','scaler_params')),
                   *sorted((ROOT/'AAAdata/Appendix_3/对齐版本').glob('*.pkl'))]},
        'code':{str(f.relative_to(ROOT)):digest(f) for f in sorted(list((ROOT/'problem2').glob('*.py'))+
                 list((ROOT/'tests').glob('test_*.py'))+[Path(__file__).resolve()])}}
    manifest=root/'pipeline.json'
    if root.exists():
        if not args.resume:raise FileExistsError(f'{root}已存在；使用--resume或换run-name')
        saved=read(manifest)
        if saved['identity']!=identity:raise ValueError('配置、代码或正式权重已改变，请使用新run-name')
    else:
        root.mkdir(parents=True)
        saved={'identity':identity,'created_at':datetime.now(timezone.utc).isoformat(),'stages':{}}
        write(manifest,saved)
        write(root/'formal_config.json',formal|{'output':str(root/'formal'),'figures':str(root/'formal/figures')})
        write(root/'ablation_config.json',spec)
    for stage,cmd in stages:
        if saved['stages'].get(stage,{}).get('status')=='completed':
            print(f'SKIP 已完成：{stage}',flush=True);continue
        saved['stages'][stage]={'status':'running','command':cmd};write(manifest,saved)
        try:execute(stage,cmd,root)
        except BaseException as exc:
            saved['stages'][stage].update(status='failed',error=str(exc));write(manifest,saved);raise
        saved['stages'][stage].update(status='completed',completed_at=datetime.now(timezone.utc).isoformat())
        write(manifest,saved)
    if digest(checkpoint)!=identity['formal_checkpoint_sha256']:raise RuntimeError('正式权重在运行期间发生变化')
    saved.update(status='completed',formal_checkpoint_unchanged=True);write(manifest,saved)
    lines=['# 问题二完整运行结果','','所有选定阶段成功。',f'- 模式：{args.mode}',f'- 训练seed：{spec["seeds"]}',
           f'- 小样本检查：{args.smoke}','- 正式模型权重未改变。','- 阶段状态和复现参数：pipeline.json；日志：logs/。']
    if args.mode in ('full','formal'):
        score=read(root/'formal/test_full.json');lines += [f'- 正式模型test ACC：{score["accuracy"]:.2%}',
           '- 完整/缺失预测、valid场景与附件3结果：formal/。']
    if args.mode in ('full','ablation'):
        lines += [f'- 消融结果与六张PNG/PDF图：../../ablations/{name}/。',
                  '- 消融研究组不会自动替换正式模型。']
    (root/'运行结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'\n全部完成：{root}/运行结果.md',flush=True)


if __name__=='__main__':main()
