"""
This 
"""
import torch


from datasets import Dataset
from collections import defaultdict
from typing import List, Dict
from transformers import AutoTokenizer, DistilBertModel
from nltk.stem import WordNetLemmatizer

import src.text_processing.preprocess as pr

bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")




def _prepare_sentence(sentence, stop_words_set: set) -> List[int]:
    """This function tokenizes, lemmatizes, removes stop words 

    Args:
        sentence (_type_): _description_
        stop_words_set (set): _description_

    Raises:
        ValueError: _description_

    Returns:
        List[int]: _description_
    """

    if not isinstance(stop_words_set, set):
        raise ValueError(f"The stop words must be a set. \nFound: {type(stop_words_set)}")
    lem = WordNetLemmatizer()
    # first tokenizer with 'bert_tokenizer'
    return bert_tokenizer.convert_tokens_to_ids([lem(t) for t in bert_tokenizer(sentence) if t not in stop_words_set])

def _process_row(source_txt, target_txt, counter, stop_words_set):
    source_ids, target_ids = _prepare_sentence(source_txt, stop_words_set), _prepare_sentence(target_txt, stop_words_set)   

    for i in source_ids:
        counter[i]["source"] += 1
    
    for i in target_ids:
        counter[i]["target"] += 1


def _toxic_unigram_batch(batch: Dict, counter, stop_words):
    _ = [_process_row(source_txt=v['source'], target_txt=v['target'], counter=counter, stop_words_set=stop_words) for _, v in batch.items()]

    return batch

def build_unigram_counter(dataset: Dataset):
    # apply the _toxic_unigram_batch function on the 'dataset' object
    stopwords = pr.standard_stop_words()  
    counter = defaultdict(lambda : {"source": 0, "target": 0})
    dataset.map(lambda b: _toxic_unigram_batch(b, counter=counter, stop_words=stopwords), batched=True)