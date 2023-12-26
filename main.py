"""
AdaDrop PyTorch implementation

AdaDrop:    introduces the gradient statistics during back propagation to
            influence the probability of dropping a node rather than naively
            sampling from a uniform distribution in traditional Dropout


Credits: PyTorch training setup was heavily inspired by the works in: https://github.com/pytorch/examples/blob/main/mnist/main.py#L122

Author(s): Ali Shirazi-Nejad (University of Texas at Arlington, Vision-Learning-Mining Lab)
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import einops

import numpy as np
import random

SEED = 0

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=10)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')
        x = self.dropout(x)
        output = self.fc(x)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}")

            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    total_test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {total_test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset)}%)\n")


def main():
    parser = argparse.ArgumentParser(description="AdaDrop Training and Testing")
    parser.add_argument('--batch_size', type=int, default=256, )


if __name__ == "__main__":
    main()