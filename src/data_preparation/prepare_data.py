""" 
This script contains functionalities to load and save datasets.
"""
import os
import random

import pandas as pd
import numpy as np

from typing import Union, Tuple, Dict
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import set_seed

from src.text_processing import preprocess as pr
# set the random seeds across different modules to the save value
np.random.seed(69)
random.seed(69)
set_seed(seed=69)


def load_aux_data():
    # set the seed for reproducibility, set_seed(69)
    aux_data = load_dataset('s-nlp/paradetox')['train'] # the dataset has only one split: 'train'
    # shuffle the data
    aux_data = aux_data.rename_column('en_toxic_comment', 'source').rename_column('en_neutral_comment', 'target').shuffle(seed=69)
    
    return aux_data


def fix_initial_data(initial_data_tsv: Union[Path, str]):
    base_name, ext = os.path.splitext(os.path.basename(initial_data_tsv))

    if base_name != 'filtered' or ext != '.tsv':
        raise ValueError(f"The initial data is expected to be saved in a file named 'filtered.tsv'")

    df = pd.read_csv(initial_data_tsv, index_col=0, sep='\t')

    # define a function that will fix the target and source sentences in row in the dataframe
    def fix_order_map(row):
        row['source'], row['target'] = (row['reference'], row['translation']) if row['ref_tox'] > row['trn_tox'] else (row['translation'], row['reference'])
        row['source_tox'], row['target_tox'] = (row['ref_tox'], row['trn_tox']) if row['ref_tox'] > row['trn_tox'] else (row['trn_tox'], row['ref_tox'])
        return row

    # fix the data
    df_fixed = df.apply(fix_order_map, axis=1)
    df_fixed.drop(columns=['translation', 'reference', 'ref_tox', 'trn_tox'], inplace=True)
    assert np.all(df_fixed['source_tox'] > df_fixed['target_tox'])
    
    # save the fixed data as csv
    data_folder = Path(initial_data_tsv).parent
    df_fixed.to_csv(os.path.join(data_folder, 'fixed.csv'), index=False)


def prepare_all_data(fixed_data_file: Union[Path, str]):    
    base_name, ext = os.path.splitext(os.path.basename(fixed_data_file))

    if base_name != 'fixed' or ext != '.csv':
        raise ValueError(f"The initial data is expected to be saved in a file named 'fixed.tsv'")
    
    # load the data as a Dataset object
    original_data = load_dataset('csv', data_files=fixed_data_file, split='train')
    original_data = original_data.remove_columns(['source_tox', 'target_tox', 'similarity', 'lenght_diff'])
    
    # load the auxiliary dataset
    aux_data = load_aux_data() 

    # time to concatenate the datasets 
    all_data = concatenate_datasets([original_data, aux_data])
    
    all_data = all_data.shuffle(seed=69)
    # make sure to 
    # save the data

    data_folder = Path(fixed_data_file).parent    
    all_data.to_csv(os.path.join(data_folder, 'all_data.csv'), index=False, sep=',')


def data_split(all_data: Union[Dataset, Path, str],
               train_portion: float = 0.96, 
               val_portion: float = 0.02
               ) -> Tuple[Dataset, Dataset, Dataset]:
    # read the data
    data = pd.read_csv(all_data) if isinstance(all_data, (Path, str)) else all_data

    if train_portion + val_portion >= 1: 
        raise ValueError(f"The test portion cannot be zero.") 

    n1 = int(len(data) * train_portion)
    n2 = int(len(data) * val_portion)

    train_data = data.select(range(len(data) - n1, len(data)))
    val_data = data.select(range(n2))
    test_data = data.select(range(n2 + 1, len(data) - n1))

    return train_data, val_data, test_data
    

def _remove_nes_batch(batch: Dict):
    """
    this function is designed to be called by the 'map' function. This function will convert 
    """
    new_batch = dict([(k, pr.uniform_ne_batched(v)) for k, v in batch.items()])
    return new_batch    

def _process_text(text: str) -> str:
    return pr.no_extra_spaces(pr.no_extra_chars(pr.to_lower(text)))


def _process_batch(batch):
    """This function recieves  batch of samples from the original data. It returns a new batch where each
    'source' and 'target' text data will be processed using the function above
    """
    new_batch = dict([(k, [_process_text(t) for t in v]) for k, v in batch.items()])
    return new_batch


def process_text_data(data: Union[Dataset, Path, str], 
                      save_path: Union[Path, str] = None) -> Dataset:  
    
    if isinstance(data, (Path, str)):
        data = load_dataset('csv', data_files=data, split='train')

    # simply create function that carries both processing steps consecutively 
    new_data = data.map(lambda b: _process_batch(_remove_nes_batch(b)), batched=True)   

    # save the data is the path is give
    if save_path is not None:
        new_data.to_csv(save_path, index=False)
    
    return new_data