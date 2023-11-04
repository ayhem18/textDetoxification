"""This script contains the main definition of the custom toxicity loss.
"""

import torch

import src.toxicity_scores.n_grams as ng 

from typing import Dict, Union, List
from torch import nn

from src.training_utilities.pytorch_utilities import get_module_device


class ToxLoss(nn.Module):
    def __init__(self,
                 uni_gram: Dict[int, float],
                 bi_gram: Dict[int, float], 
                 ignore_map: Dict[int, float],
                 default_toxicity: float, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.uni = uni_gram
        self.bi = bi_gram
        self.ignore_map = ignore_map
        self.default_tox = default_toxicity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the first step is to convert 'x' into a numpy array
        x_device = get_module_device(x)
        x_np = x.detach().cpu().numpy()
        
        scores = torch.from_numpy(x_np.apply_along_axis(lambda row: ng.indices_toxicity_score(indices=row, 
                                                                             ignore_map=self.ignore_map, 
                                                                             uni_gram=self.uni,
                                                                             bi=self.bi))).to(x_device)

        