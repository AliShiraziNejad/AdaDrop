import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

class ScalingMethod(Enum):
    INVERSE = "inverse"
    SOFTMAX = "softmax"
    NORMALIZED = "norm"

class AdaDrop_L(nn.Module):
    def __init__(self, scaling: ScalingMethod) -> None:
        super().__init__()
        self.gradients = None
        self.scaling = scaling

    def _get_gradient_hook(self):
        def hook(grad):
            self.gradients = grad.abs()
        return hook

    def register_gradient_hook(self, layer):
        """
        Register a hook to the given layer to capture gradients.
        """
        layer.register_backward_hook(self._get_gradient_hook())

    def _calculate_dropout_probabilities(self):
        if self.scaling == ScalingMethod.INVERSE:
            normalized_gradients = self.gradients / self.gradients.sum()
            return 1 - normalized_gradients

        elif self.scaling == ScalingMethod.SOFTMAX:
            return F.softmax(self.gradients, dim=0)

        elif self.scaling == ScalingMethod.NORMALIZED:
            max_grad = self.gradients.max()
            return self.gradients / max_grad if max_grad > 0 else self.gradients

        else:
            raise ValueError(f"Invalid scaling method: {self.scaling}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradients is not None:
            dropout_probabilities = self._calculate_dropout_probabilities()
            mask = torch.bernoulli(dropout_probabilities).to(input.device)
            return input * mask
        return input
