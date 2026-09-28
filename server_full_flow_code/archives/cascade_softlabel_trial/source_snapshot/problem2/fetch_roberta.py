"""固定版本下载三分类情感资源，推理仅在本地进行。"""
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from shared.common import path, json_write


def fetch():
    repo='cardiffnlp/twitter-roberta-base-sentiment-latest'
    revision=HfApi().model_info(repo,token=False).sha
    source=snapshot_download(repo,revision=revision,token=False,
        cache_dir=str(path('AAAmodel/hf_cache')),
        allow_patterns=['config.json','pytorch_model.bin','model.safetensors',
                        'vocab.json','merges.txt','tokenizer*','special_tokens_map.json','README.md'])
    directory=path('AAAmodel/roberta-sentiment')
    model=AutoModelForSequenceClassification.from_pretrained(source,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,use_fast=True)
    assert model.config.num_labels==3 and model.config.hidden_size==768
    model.save_pretrained(directory,safe_serialization=True);tokenizer.save_pretrained(directory)
    json_write(directory/'provenance.json',{'repo':repo,'revision':revision,
        'source':f'https://huggingface.co/{repo}', 'external_training_task':'TweetEval sentiment',
        'label_order':['negative','neutral','positive'], 'local_data_used_by_fetch':False,
        'input_conversion':'retained BERT ids -> decoded text -> RoBERTa tokenizer',
        'license':'cc-by-4.0; Cardiff NLP / TimeLMs'})
    print(directory,flush=True)


if __name__=='__main__':fetch()
