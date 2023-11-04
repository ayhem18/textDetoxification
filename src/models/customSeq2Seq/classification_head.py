"""
This script contains different general classes of classification heads with different design choices: either my own
design or inspired from other sources.
"""

import torch
import numpy as np

from torch import nn
from typing import Sequence, Iterator
from torch.nn import Module
from abc import ABC, abstractmethod


class ClassificationHead(ABC, nn.Module):
    _RELU = 'relu'
    _LEAKY_RELU = 'leaky_relu'
    _TANH = 'tanh'
    _ACTIVATIONS = [_RELU, _LEAKY_RELU, _TANH]

    _ACTIVATION_MAP = {_RELU: nn.ReLU(inplace=True),
                       _TANH: nn.Tanh(),
                       _LEAKY_RELU: nn.LeakyReLU(inplace=True)}

    @classmethod
    def linear_block(cls,
                     input_features: int,
                     output_features: int,
                     is_final: bool = False,
                     activation: str = 'leaky_relu',
                     *args
                     ) -> nn.Sequential:
        """
        create a linear block with batch normalization, and activation
        Args:
            input_features: the number of features passed to the linear block
            output_features: the number of features expected from the linear block
            is_final: determines whether the batch normalization and activation function should be applied
            activation: a string that represents the activation applied
            *args: additional arguments to pass to the activation function
        Returns: a nn.
        """
        components = [nn.Linear(in_features=input_features,
                                out_features=output_features)]
        if activation == cls._RELU:
            activation = nn.ReLU(inplace=True)

        elif activation == cls._LEAKY_RELU:
            activation = nn.LeakyReLU(*args, inplace=True)
        else:
            activation = nn.Tanh(inplace=True)

        components.extend([] if is_final else [nn.BatchNorm1d(output_features), activation])

        return nn.Sequential(*components)

    # all classifiers should have the 'num_classes' and 'in_features' attributes
    def __init__(self, num_classes: int,
                 in_features: int,
                 activation='leaky_relu'):
        super().__init__()
        # take into account the case of binary-classification
        self._num_classes = num_classes if num_classes > 2 else 1
        self._in_features = in_features
        self._activation = activation
        # the actual mode that does the heavy lifting
        self.classifier = None

    @property
    def num_classes(self):
        return self._num_classes

    # a setter for num_classes and in_features
    @num_classes.setter
    def num_classes(self, x: int):
        self._num_classes = x if x > 2 else 1
        self._build_classifier()

    @property
    def in_features(self):
        return self._in_features

    @in_features.setter
    def in_features(self, x: int):
        self._in_features = x
        self._build_classifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier.forward(x)

    def children(self) -> Iterator[nn.Module]:
        return self.classifier.children()

    def named_children(self) -> Iterator[tuple[str, Module]]:
        return self.classifier.named_children()

    def modules(self) -> Iterator[nn.Module]:
        return self.classifier.modules()

    @abstractmethod
    def _build_classifier(self):
        # this function represents the main design of the classification head
        pass


class ExponentialClassifier(ClassificationHead):
    def __init__(self,
                 num_classes: int,
                 in_features: int,
                 num_layers: int,
                 activation: str = 'leaky_relu'):
        # the usual parent's class call
        super().__init__(num_classes, in_features, activation)
        self.num_layers = num_layers
        self.output = 1 if num_classes == 2 else num_classes
        self._build_classifier()

    def _build_classifier(self):
        base_power = int(np.log2(self.in_features))
        powers = np.linspace(start=int(np.log2(self.output)), stop=base_power, num=self.num_layers)
        # make sure to convert to integers
        num_units = [int(2 ** p) for p in powers][::-1]
        # set the last element to the actual number of classes
        num_units[-1] = self.output
        num_units = [self.in_features] + num_units

        blocks = [self.linear_block(input_features=num_units[i],
                                    output_features=num_units[i + 1],
                                    is_final=False,
                                    activation=self._activation) for i in range(len(num_units) - 2)]

        # add the last layer by setting the 'is_final' argument to True
        blocks.append(self.linear_block(input_features=num_units[-2],
                                        output_features=num_units[-1],
                                        is_final=True))

        self.classifier = nn.Sequential(*blocks)


class GenericClassifier(ClassificationHead):
    """
    This is a generic classifier where the architecture is fully defined by the input.
    The user simply indicates the initial_input, and the number of classes, as well as the number of hidden units.
    """
    def __init__(self,
                 num_classes: int,
                 in_features: int,
                 hidden_units: Sequence[int] = None,
                 activation: str = 'leaky_relu',
                 *args, **kwargs):
        super().__init__(num_classes, in_features, activation)

        if hidden_units is None:
            hidden_units = []

        if num_classes < 2:
            raise ValueError('The number of classes cannot be less than 2.\n'
                             f'FOUND: {num_classes}')

        self.hidden_units = hidden_units
        self._build_classifier()

    def _build_classifier(self) -> None:

        num_units = [self.in_features] + self.hidden_units + [self.num_classes]

        blocks = [self.linear_block(input_features=num_units[i],
                                    output_features=num_units[i + 1],
                                    is_final=False,
                                    activation=self._activation) for i in range(len(num_units) - 2)]

        # add the last block as final
        blocks.append(self.linear_block(input_features=num_units[-2],
                                        output_features=num_units[-1],
                                        is_final=True))

        self.classifier = nn.Sequential(*blocks)
