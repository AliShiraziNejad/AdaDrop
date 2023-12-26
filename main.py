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
from torch.utils.data import DataLoader

from AdaDrop import AdaDrop

import einops

import numpy as np
import random


class CNN_noDO(nn.Module):
    def __init__(self):
        super(CNN_noDO, self).__init__()
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

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output


class CNN_regDO(nn.Module):
    def __init__(self):
        super(CNN_regDO, self).__init__()
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

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        output = self.fc3(x)
        return output


class CNN_AdaDrop_inverse(nn.Module):
    def __init__(self):
        super(CNN_AdaDrop_inverse, self).__init__()
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

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.adaDrop1 = AdaDrop(self.fc1, scaling="inverse", N=1)
        self.adaDrop2 = AdaDrop(self.fc2, scaling="inverse", N=1)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')
        x = self.AdaDrop(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.AdaDrop(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.AdaDrop(x)
        output = self.fc3(x)
        return output


class CNN_AdaDrop_softmax(nn.Module):
    def __init__(self):
        super(CNN_AdaDrop_softmax, self).__init__()
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

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.adaDrop1 = AdaDrop(self.fc1, scaling="softmax", N=1)
        self.adaDrop2 = AdaDrop(self.fc2, scaling="softmax", N=1)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')
        x = self.AdaDrop(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.AdaDrop(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.AdaDrop(x)
        output = self.fc3(x)
        return output


class CNN_AdaDrop_norm(nn.Module):
    def __init__(self):
        super(CNN_AdaDrop_norm, self).__init__()
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

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.adaDrop1 = AdaDrop(self.fc1, scaling="norm", N=1)
        self.adaDrop2 = AdaDrop(self.fc2, scaling="norm", N=1)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')
        x = self.AdaDrop(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.AdaDrop(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.AdaDrop(x)
        output = self.fc3(x)
        return output


def train(args, model, device, train_loader, optimizer, epoch, log_file):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        output = model(inputs)
        loss = F.nll_loss(output, labels)

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            log_file.write(
                f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}\n")

            if args.dry_run:
                break


def test(model, device, testloader, log_file):
    model.eval()
    total_test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total_test_loss /= len(testloader.dataset)

    log_file.write(
        f"\nTest set: Average loss: {total_test_loss}, Accuracy: {correct}/{len(testloader.dataset)} ({100.0 * correct / len(testloader.dataset)}%)\n")


def main():
    valid_datasets = {"MNIST", "CIFAR10"}

    valid_models = {
        "CNN_noDO": CNN_noDO(),
        "CNN_regDO": CNN_regDO(),
        "CNN_AdaDrop_inverse": CNN_AdaDrop_inverse(),
        "CNN_AdaDrop_softmax": CNN_AdaDrop_softmax(),
        "CNN_AdaDrop_norm": CNN_AdaDrop_norm()
    }

    parser = argparse.ArgumentParser(description="AdaDrop Training and Testing")

    parser.add_argument('--dataset', type=str, metavar='DS',
                        required=True, help='dataset for training and evaluation')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--test_batch_size', type=int, default=1024, metavar='N',
                        help='input batch size for testing (default: 1024)')

    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')

    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')

    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--no_mps', action='store_true', default=False,
                        help='disables macOS GPU training')

    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()

    assert args.dataset in valid_datasets, "Failed to provide valid dataset: MNIST, CIFAR10"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.MNIST(root='./data',
                                  train=True,
                                  download=True,
                                  transform=transform)
        testset = datasets.MNIST(root='./data',
                                 train=False,
                                 download=True,
                                 transform=transform)

    elif args.dataset == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(root='./data',
                                    train=True,
                                    download=True,
                                    transform=transform)
        testset = datasets.CIFAR10(root='./data',
                                   train=False,
                                   download=True,
                                   transform=transform)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    for model_name, model in valid_models.items():
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        with open(f"{model_name}_log.log", "w") as log_file:
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, trainloader, optimizer, epoch, log_file)
                test(model, device, testloader, log_file)
                scheduler.step()

            if args.save_model:
                torch.save(model.state_dict(), f"{args.dataset}_{model_name}.pt")


if __name__ == "__main__":
    main()
