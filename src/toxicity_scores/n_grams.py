"""
This 
"""
import os
import pickle

from datasets import Dataset
from collections import defaultdict
from typing import List, Dict, Union, Iterable
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
    uni_gram, bi_gram = dict(uni_gram), dict(bi_gram)

    with open(os.path.join(save_folder, 'uni_gram.pkl'), 'wb') as f:     
        # the default is the highest protocol. Let's leave it at that
        pickle.dump(uni_gram, f)

    with open(os.path.join(save_folder, 'bi_gram.pkl'), 'wb') as f:     
        # the default is the highest protocol. Let's leave it at that
        pickle.dump(bi_gram, f)

    return uni_gram, bi_gram


## FUNCTIONS FOR CALCULATING THE TOXICITY OF A SEQUENCE OF TOKENS IDS 

def build_ignore_toxic_map(default_toxicitity: float, stop_words, save_folder: Union[str, Path]=None) -> Dict[str, float]:
    ignore = {}

    # all special token should be associated with "0" toxicity (except the UNK loss)
    for tk_id in bert_tokenizer.all_special_ids:
        ignore[tk_id] = (0 if tk_id != bert_tokenizer.unk_token_id else default_toxicitity)
    
    # iterate through the tokenizer's vocabulary, and if the tokens have punctuation, set their toxicity score to '0'
    non_alpha_counter = 0
    for text, i in bert_tokenizer.vocab.items():
        if not text.isalpha():
            non_alpha_counter += 1
            ignore[i] = 0

    print(f"{non_alpha_counter} non alpha tokens")

    sw_indices = bert_tokenizer.convert_tokens_to_ids(list(stop_words))

    for i in sw_indices:
        ignore[i] = default_toxicitity
        
    save_folder =  os.path.dirname(os.path.realpath(__file__)) if save_folder is None else save_folder
    with open(os.path.join(save_folder, 'ignore_map.pkl'), 'wb') as f:     
        # the default is the highest protocol. Let's leave it at that
        pickle.dump(ignore, f)

    return ignore

def indices_toxicity_score(indices: Iterable[int], ignore_map:Dict, uni_gram: Dict, bi_gram: Dict, default_toxicity: float):
    ignore_indices, toxic_indices = ([i for i, index in enumerate(indices) if index in ignore_map], 
                                     [i for i, index in enumerate(indices) if index not in ignore_map])

    # default for bigram is 0, default for unigram is the passed value
    bi_gram.setdefault(0)
    uni_gram.setdefault(default_toxicity)

    result = [0 for _ in indices]

    for i in ignore_indices:
        result[i] += ignore_map[indices[i]] 
    
    for i in toxic_indices:
        result[i] += (uni_gram[indices[i]] if indices[i] in uni_gram else 0) 

    # add the bi-gram toxicity
    for j in range(len(toxic_indices) - 1):
        i1, i2 = indices[toxic_indices[j]], indices[toxic_indices[j + 1]]
        tox_score = (bi_gram[(i1, i2)] if (i1, i2) in bi_gram else 0) 
        result[j] += 2 * tox_score
        result[j + 1] += 2 * tox_score

    return result
     



