"""
This script contains functionalities needed to estimate the toxicity score of each uni / bi -gram.
"""

import os
import pickle

from datasets import Dataset
from collections import defaultdict
from typing import List, Dict, Union, Iterable, Set
from pathlib import Path
from transformers import AutoTokenizer, DistilBertModel
from nltk.stem import WordNetLemmatizer

import src.text_processing.preprocess as pr

bert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")


### FUNCTIONS FOR CREATING THE N-GRAMS

def _prepare_sentence(sentence: str, stop_words_set: set) -> List[int]:
    """This function tokenizes, lemmatizes, removes stop words and return the tokens' ids by the Bert-distilled tokenizer. 
    """
    if not isinstance(stop_words_set, set):
        raise ValueError(f"The stop words must be a set. \nFound: {type(stop_words_set)}")
    lem = WordNetLemmatizer()
    # nested calls were a design choice to ensure the least amount of overhead.
    result =  [lem.lemmatize(t) for t in pr.tokenize(sentence) if t.isalpha() and t not in stop_words_set]
    return result

def _toxic_unigram_row(source_txt, target_txt, uni_gram, bi_gram, stop_words_set):
    """Given source (toxic) and target (neutral) texts as well as the maps and the list of stop words, This function updates the
    maps (unigram to toxicity score) and (bi-gram to toxicity) with the results of the pair of given sentences

    """
    # prepare the data
    source_ids, target_ids = _prepare_sentence(source_txt, stop_words_set), _prepare_sentence(target_txt, stop_words_set)   

    # add the uni grams
    for i in source_ids:
        uni_gram[i]["source"] += 1
    
    for i in target_ids:
        uni_gram[i]["target"] += 1

    # work on the bi-grams
    for i in range(len(source_ids) - 1):
        bi_gram[(source_ids[i], source_ids[i + 1])]["source"] += 1

    for i in range(len(target_ids) - 1):
        bi_gram[(target_ids[i], target_ids[i + 1])]["target"] += 1


def _toxic_unigram_batch(batch: Dict, uni_gram, bi_gram, stop_words):
    # batch is expected to be a dictionary with fields: 
    # 1. 'source': batched source sentences
    # 2. 'text': batched target sentences 

    # write the code as list comprehension for computational efficiency.
    _ = [_toxic_unigram_row(source_txt=s, target_txt=t, 
                            uni_gram=uni_gram, 
                            bi_gram=bi_gram, 
                            stop_words_set=stop_words) for s, t in zip(batch['source'], batch['target'])]
    return batch


def build_unigram_counter(dataset: Dataset, save_folder: Union[str, Path] = None):
    # apply the _toxic_unigram_batch function on the 'dataset' object
    stopwords = pr.standard_stop_words()  
    uni_gram = defaultdict(lambda : {"source": 0, "target": 0})
    bi_gram = defaultdict(lambda: {"source": 0, "target": 0})

    dataset.map(lambda b: _toxic_unigram_batch(b, uni_gram=uni_gram, bi_gram=bi_gram, stop_words=stopwords), batched=True)
    
    save_folder =  os.path.dirname(os.path.realpath(__file__)) if save_folder is None else save_folder
    # convert the default dict to a standard Dictionary before serializing it
    uni, nbi = dict(uni_gram), dict(bi_gram)

    # make sure to convert it to the values needed
    u, b = dict([(k, (v["source"] + 1) / (v["target"] + 1)) for k, v in uni.items()]) , dict([(k, (v["source"] + 1) / (v["target"] + 1)) for k, v in bi_gram.items()])  

    with open(os.path.join(save_folder, 'uni_gram.obj'), 'wb') as f:     
        # the default is the highest protocol. Let's leave it at that
        pickle.dump(u, f)

    with open(os.path.join(save_folder, 'bi_gram.obj'), 'wb') as f:     
        # the default is the highest protocol. Let's leave it at that
        pickle.dump(b, f)

    return u, b

def get_toxicity_attributes(sentence: str, 
                            uni_gram: Dict, 
                            bi_gram: Dict,
                            uni_threshold: float = 0.099, 
                            bi_threshold: float = 0.078,
                            default_toxicity: float = 0.02, 
                            min_num_words: int = 1) -> set[str]:
    """_summary_

    Args:
        sentence (str): input sentence
        uni_gram (Dict): The map between unigrams and toxicity scores
        bi_gram (Dict): the map between bi-grams and toxicity scores
        uni_threshold (float, optional): The threshold for a uni gram to be a toxicity attribute. Defaults to 0.099. 
        (40% percentil of all toxicity scores)
        bi_threshold (float, optional): The threshold for a bi-gram to be a toxicity attribute. Defaults to 0.078.
        (40% percentil of all bigram toxicity scores)
        default_toxicity (float, optional): The default. Defaults to 0.02.
        (20% percentile of all uni-gram toxicity)
        min_num_words (int, optional): The number of uni grams returned in case no toxicity attribute was detected . 
        Defaults to 1.

    Returns:
        set[str]: _description_
    """
    #1.prepare the sentence
    s = _prepare_sentence(sentence, pr.standard_stop_words())
    toxic_attributes_uni = set()
    toxic_attributes_bi = set()

    for t in s: 
        tox = uni_gram[t] if t in uni_gram else default_toxicity
        if tox >= uni_threshold:
            toxic_attributes_uni.add(t)
    
    for i in range(len(s) - 1):
        bi = (s[i], s[i + 1])
        tox = bi_gram[bi] if bi in  bi_gram else default_toxicity
        
        if tox >= bi_threshold:
            toxic_attributes_uni.update(bi)
    
    # convert all tokens into a single set
    toxic_attributes_bi.update(toxic_attributes_uni)
    
    # if no specific word was considered as a toxicity attribute, we will simply consider the most toxic unigram
    if len(toxic_attributes_bi) == 0:
        tokens_sorted = sorted(s, key=lambda t: uni_gram[t], reverse=True)
        return set(tokens_sorted[:min_num_words])

    return toxic_attributes_bi

     