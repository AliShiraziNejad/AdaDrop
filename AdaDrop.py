import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class AdaDropFunction(Function):
    @staticmethod
    def forward(ctx, input, drop_prob):
        mask = torch.bernoulli(1 - drop_prob)
        ctx.save_for_backward(mask)
        output = input * mask
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = grad_output * mask
        return grad_input, None



class AdaDrop(nn.Module):
    def __init__(self, layer, scaling='inverse', N=5):
        super(AdaDrop, self).__init__()
        self.layer = layer
        self.scaling = scaling
        self.N = N
        self.gradients = []
        self.layer.register_full_backward_hook(self.gradient_hook)

    def gradient_hook(self, grad_output):
        grad = grad_output[0]
        self.gradients.append(grad)
        if len(self.gradients) > self.N:
            self.gradients.pop(0)

    def get_drop_prob(self, gradients):
        """
        Calculates dropout probabilities for nodes in a neural network layer based on the statistics of their gradients.
        This method averages the gradients over a specified number of backward passes and applies a scaling method to
        determine the probability of each node being dropped. The aim is to adaptively drop nodes that contribute less
        to the model's learning, based on the historical behavior of their gradients.

        Parameters:
        - gradients (List[torch.Tensor]): A list of gradient tensors from the last N backward passes. Each tensor in the list
          should have the same shape, corresponding to the shape of the layer's output that this adaDrop is applied to.
          These gradients are stacked and averaged to provide a single gradient tensor that represents the average
          gradient across the backward passes.

        Returns:
        - torch.Tensor: A tensor of the same shape as each gradient tensor, containing the dropout probabilities for each
          node in the layer. Each element of the tensor represents the probability that the corresponding node will be
          dropped during the forward pass.

        Raises:
        - ValueError: If an unsupported scaling method is specified. The supported scaling methods are "inverse",
          "softmax", and "norm".

        The scaling methods:
        - "inverse": Scales the gradients inversely proportional to their magnitude sum. Nodes with smaller average
          gradients over N passes have a higher probability of being kept.
        - "softmax": Applies a softmax function to the gradients, turning them into probabilities directly. This method
          considers both the magnitude and distribution of gradients.
        - "norm": Normalizes the gradients by the maximum gradient magnitude. This scaling gives a relative sense of
          gradient importance, with the highest gradient normalized to 1.
        """

        gradients = torch.stack(gradients).mean(0)

        if self.scaling == "inverse":
            normalized_gradients = gradients / gradients.sum()
            drop_prob = 1 - normalized_gradients

        elif self.scaling == "softmax":
            drop_prob = F.softmax(gradients, dim=0)

        elif self.scaling == "norm":
            max_grad = gradients.max()
            drop_prob = gradients / max_grad if max_grad > 0 else gradients

        else:
            raise ValueError(f"Unsupported scaling method: '{self.scaling}'")

        return drop_prob

    def forward(self, x):
        if self.training:
            drop_prob = self.get_drop_prob(self.gradients)
            return AdaDropFunction.apply(self.layer(x), drop_prob)
        else:
            return self.layer(x)
