"""This script contains the functionalities needed to run the baseline solution
"""


import os, sys
from pathlib import Path
from typing import Dict, Tuple, List
import spacy
import random
import pickle

import pandas as pd

from datasets import load_dataset, Dataset
from empiricaldist import Cdf
from transformers import pipeline, AutoTokenizer

HOME = os.getcwd()
current = HOME 
while 'src' not in os.listdir(current):
    current = Path(current).parent

PARENT_DIR = current
DATA_FOLDER = os.path.join(PARENT_DIR,  'data')
sys.path.append(str(current))
sys.path.append(os.path.join(str(current), 'data_analysis'))
sys.path.append(os.path.join(str(current), 'evaluation'))
sys.path.append(os.path.join(str(current), 'text_processing'))

import src.text_processing.preprocess as pr
import src.data_preparation.prepare_data as prd
from src.models.baseline import n_grams as ng

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def process_text(text: str) -> str:
    return pr.no_extra_spaces(pr.no_extra_chars(pr.to_lower(text)))

def process_batch(batch: Dict, nlp):
    p = random.random()
    if p < 10 ** -5:
        print("really ?")
    return dict([(k, [process_text(t) for t in v]) for k, v in ({"source": pr.uniform_ne_batched(batch['source'], nlp), 
                                                                 "target": pr.uniform_ne_batched(batch['target'], nlp)}).items()]) 



def build_maps(processed_data: Dataset):
    """this can be seed as the 'training' phase of the baseline model. 
    The maps between unigrams and bi-grams are built 

    Args:
        processed_data Dataset, the processed dataset

    Returns:
        _type_: The maps and the thresholds needed for the baseline approach
    """


    uni, bi = ng.build_unigram_counter(processed_data)
    u, b = (dict([(k, (v["source"] + 1) / (v["target"] + 1)) for k, v in uni.items()]) , 
    dict([(k, (v["source"] + 1) / (v["target"] + 1)) for k, v in bi.items()]))  
    # find the thresholds

    u_scores = [v for k, v in u.items()]
    bi_scores = [v for k, v in b.items()]

    cdf_u = Cdf.from_seq(u_scores)
    cdf_b = Cdf.from_seq(bi_scores)

    toxicity_threshold_u = cdf_u.forward(0.4).item()
    toxicity_threshold_bi = cdf_b.forward(0.4).item()
    default_toxicitiy = cdf_u.forward(0.2).item()
    toxicity_threshold_bi, toxicity_threshold_u, default_toxicitiy

    return u, b, toxicity_threshold_u, toxicity_threshold_bi, default_toxicitiy





if __name__ == '__main__':
    try: 
        data = prd.prepare_all_data(fixed_data_file=os.path.join(DATA_FOLDER, 'fixed.csv'), save=False)
    except FileNotFoundError:
        print("PLEASE MAKE SURE TO RUN THE DATA SCRIPS BEFORE RUNNING THE MODEL SCRIPTS!!\n ABORTING !!")

    all_processed_data_path = os.path.join(DATA_FOLDER, 'all_data_processed.csv')

    # either prepare the data or load it    
    if not os.path.exists(all_processed_data_path):    
        # load the nlp object in advnace
        nlp = spacy.load("en_core_web_sm")
        processed_data = data.map(lambda b: process_batch(b, nlp), batched=True)
    else:
        processed_data = load_dataset("csv", data_files=os.path.join(DATA_FOLDER, "all_data_processed.csv"), split='train')
        processed_data = processed_data.filter(lambda s: (isinstance(s['source'], str) and isinstance(s['target'], str))) 

    uni_path = os.path.join(SCRIPT_DIR, 'uni_gram.obj')
    bi_path = os.path.join(SCRIPT_DIR, 'bi_gram.obj')

    toxicity_threshold_u, toxicity_threshold_bi, default_toxicitiy = None, None, None

    if os.path.exists(uni_path) and os.path.exists(bi_path):
        u, b = pickle.load(open(uni_path), 'rb')
    else:
        u, b, toxicity_threshold_u, toxicity_threshold_bi, default_toxicitiy = build_maps(processed_data)

    test_data = pd.read_csv(os.path.join(DATA_FOLDER, 'test_splits.csv'), nrows=10)


    ckpnt = "distilbert-base-uncased"
    mask_filler = pipeline("fill-mask", ckpnt)
    # bert = AutoModelForMaskedLM.from_pretrained(ckpnt)
    bert_tokenizer = AutoTokenizer.from_pretrained(ckpnt)
    bert_tokenizer.mask_token

    
    
    
