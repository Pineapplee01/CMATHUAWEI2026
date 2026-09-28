"""从保存的JSON重新核验解释的数值完整性。"""
import json
import numpy as np
from shared.common import path, json_write


def verify_cards(cfg):
    directory = path(cfg['explanation_output']) / 'explanations'
    files = sorted(directory.glob('*.json'))
    if len(files)!=20: raise AssertionError('附件4必须保留20张解释卡')
    errors=[]
    for file in files:
        card=json.loads(file.read_text())
        ledger=card['ledger']
        score=np.asarray(ledger['bias'])+np.asarray(ledger['single']).sum((1,2))[0]+np.asarray(ledger['pair']).sum((1,2))[0]
        if 'triple' in ledger: score += np.asarray(ledger['triple']).sum(1)[0]
        if 'text_prior' in ledger: score += np.asarray(ledger['text_prior'])[0]
        if 'global_fusion' in ledger: score += np.asarray(ledger['global_fusion'])[0]
        intensity=3*np.tanh(score[3])
        probability=np.exp(score[:3]-score[:3].max());probability/=probability.sum()
        original=np.asarray(card['coalition_outputs']['7'])
        phi=np.asarray(card['source_shapley'])
        residual=np.max(np.abs(phi.sum(0)-(original-np.asarray(card['coalition_outputs']['0']))))
        error=max(abs(intensity-card['intensity']),abs(probability[card['class']]-original[0]),float(residual))
        if error>2e-5: raise AssertionError(f'{file}: 数值重建误差 {error}')
        if int(score[:3].argmax())!=card['class']: raise AssertionError('账本预测类不一致')
        errors.append(error)
    report={'cards':len(files),'max_reconstruction_error':max(errors),'passed':True}
    json_write(directory.parent/'explanation_validation.json',report)
    return report
