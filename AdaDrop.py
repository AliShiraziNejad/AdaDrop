import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

class AdaDrop(nn.Module):
    def __init__(self, layer, scaling: str, N: int) -> None:
        super().__init__()
        self.recent_gradients = []
        self.max_length = N
        self.scaling = scaling
        self.valid_scalings = ["inverse", "softmax", "norm"]
        self._register_gradient_hook(layer)

    def _get_gradient_hook(self):
        def hook(grad_input):
            grad = grad_input[0].abs()
            self.recent_gradients.append(grad)
            if len(self.recent_gradients) > self.max_length:
                self.recent_gradients.pop(0)
        return hook

    def _register_gradient_hook(self, layer):
        layer.register_backward_hook(self._get_gradient_hook())

    def _calculate_dropout_probabilities(self):
        if not self.recent_gradients:
            return None

        gradients = reduce(torch.stack(self.recent_gradients), 'n c -> c', 'mean')

        if self.scaling == "inverse":
            normalized_gradients = gradients / gradients.sum()
            return 1 - normalized_gradients

        elif self.scaling == "softmax":
            return F.softmax(gradients, dim=0)

        elif self.scaling == "norm":
            max_grad = gradients.max()
            return gradients / max_grad if max_grad > 0 else gradients

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            dropout_probabilities = self._calculate_dropout_probabilities()
            if dropout_probabilities is not None:
                mask = torch.bernoulli(dropout_probabilities).to(input.device)
                return input * mask
        return input