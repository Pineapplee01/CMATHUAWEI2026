"""获取同尺寸、同词表的SST-2情感初始化，保留来源和固定提交版本。"""
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from shared.common import path, json_write, fingerprint


def fetch():
    repo='textattack/bert-base-uncased-SST-2'
    revision='95f0f6f859b35c8ff0863ae3cd4e2dbc702c0ae2'
    source=snapshot_download(repo,revision=revision,token=False,
        cache_dir=str(path('AAAmodel/hf_cache')),
        allow_patterns=['config.json','pytorch_model.bin','model.safetensors',
                        'vocab.txt','tokenizer*','special_tokens_map.json'])
    directory=path('AAAmodel/bert-base-uncased-sst2')
    model=AutoModel.from_pretrained(source,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,use_fast=True)
    original=AutoTokenizer.from_pretrained(path('AAAmodel/bert-base-uncased'),local_files_only=True)
    if tokenizer.get_vocab()!=original.get_vocab(): raise AssertionError('SST-2骨干词表不一致')
    if model.config.hidden_size!=768 or model.config.num_hidden_layers!=12:
        raise AssertionError('骨干容量改变')
    model.save_pretrained(directory,safe_serialization=True)
    tokenizer.save_pretrained(directory)
    json_write(directory/'provenance.json',{'repo':repo,'revision':revision,
        'source':'https://textattack.readthedocs.io/en/latest/3recipes/models.html',
        'external_task':'SST-2 binary sentiment; classification head discarded',
        'architecture':'BERT base uncased, 12 layers, width 768',
        'vocabulary_matches_original':True,'sha256':fingerprint(directory),
        'local_mosei_train_or_valid_used_by_fetch':False})
    print(f'Saved {directory}',flush=True)


if __name__=='__main__': fetch()
