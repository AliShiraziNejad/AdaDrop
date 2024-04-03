import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import einops


class AdaDropFunction(Function):
    @staticmethod
    def forward(ctx, input, drop_prob):
        mask = torch.bernoulli(1 - drop_prob.to(input.device))
        # mask = torch.bernoulli(drop_prob.to(input.device))
        print(f'(TOTAL: {mask.shape[1]}) Number of units dropped: {(mask == 0).sum().item()}')
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

    def gradient_hook(self, module, grad_input, grad_output):
        grad = grad_output[0]
        self.gradients.append(grad)
        if len(self.gradients) > self.N:
            self.gradients.pop(0)

    def get_drop_prob(self, gradients):
        if len(gradients) == 0:
            return torch.zeros((1, 1), requires_grad=False)

        gradients = einops.reduce(torch.stack(gradients), 'N ... -> ...', 'mean')

        if self.scaling == "inverse":
            normalized_gradients = gradients / gradients.sum()
            drop_prob = 1 - normalized_gradients
        elif self.scaling == "softmax":
            drop_prob = F.softmax(gradients, dim=1)
            # print(drop_prob)
        elif self.scaling == "norm":
            max_grad = gradients.max()
            drop_prob = gradients / max_grad if max_grad > 0 else gradients
        else:
            raise ValueError(f"Unsupported scaling method: '{self.scaling}'")

        return drop_prob

    def forward(self, x):
        if self.training:
            drop_prob = self.get_drop_prob(self.gradients)
            return AdaDropFunction.apply(x, drop_prob)
        else:
            return self.layer(x)
