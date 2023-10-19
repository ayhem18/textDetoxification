"""
This script is written to apply the Summarization model on our problem
"""
import os.path

from typing import Union
from pathlib import Path
from torch.utils.data import Dataset


class TSVDataset(Dataset):
    def __init__(self, file_path: Union[str, Path]):
        # make sure the path ends with 'fixed.tsv'
        if not os.path.basename(file_path).endswith('fixed.tsv'):
            raise ValueError(f"The file is expected to be name 'fixed.tsv'. Found: {os.path.basename(file_path)}")

        self.path = file_path

    def __len__(self):

        return

    def __getitem__(self, idx: int):

