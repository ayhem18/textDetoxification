"""
This script contains functionalities to distill the skolkovo classifier
"""

import torch
import evaluate # a library used to compute standard metrics
import numpy as np

from typing import Iterable, Union, Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from tqdm import tqdm


