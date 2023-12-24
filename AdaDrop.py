import torch
import torch.nn as nn

class AdaDrop(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gradients = None

    def _get_gradient_hook(self):
        def hook(grad):
            self.gradients = grad.abs()
        return hook

    def register_gradient_hook(self, layer):
        """
        Register a hook to the given layer to capture gradients.
        """
        layer.register_backward_hook(self._get_gradient_hook())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and self.gradients is not None:
            # Compute probabilities inversely proportional to gradient magnitude
            normalized_gradients = self.gradients / self.gradients.sum()
            dropout_probabilities = 1 - normalized_gradients

            # Apply dropout
            mask = torch.bernoulli(dropout_probabilities).to(input.device)
            return input * mask
        return input
