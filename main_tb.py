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
import torch.optim as optim
import torch.nn.init as init
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random

from tqdm import tqdm

from models import *


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)


def train(args, model, device, train_loader, optimizer, epoch, log_file, writer):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')

    model.train()

    for batch_idx, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        output = model(inputs)
        loss = F.nll_loss(output, labels)

        loss.backward()

        optimizer.step()

        pbar.set_description(f'Epoch {epoch} Loss: {loss.item():.6f}')

        if batch_idx % args.log_interval == 0:
            log_file.write(
                f"Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.4f}%)]\tLoss: {loss.item():.4f}\n")

        global_step = (epoch - 1) * len(train_loader) + batch_idx
        writer.add_scalar('Training Loss', loss, global_step=global_step)

        if args.dry_run:
            break


def test(model, device, test_loader, log_file, epoch, writer):
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f'Testing Epoch {epoch}')

    model.eval()
    total_test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            pbar.set_description(f'Testing Epoch {epoch} Loss: {total_test_loss / (batch_idx + 1):.6f}')

    total_test_loss /= len(test_loader.dataset)

    log_file.write(
        f"\nTest set: Average loss: {total_test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset)}%)\n")

    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    writer.add_scalar('Test Loss', total_test_loss, global_step=epoch)
    writer.add_scalar('Test Accuracy', test_accuracy, global_step=epoch)


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

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')

    parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')

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

    use_cuda = args.cuda and torch.cuda.is_available()
    use_mps = args.mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device in use: {device}")

    if args.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
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
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
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
        model.apply(init_weights)
        model.to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, eta_min=0.0001, T_max=args.epochs)

        writer = SummaryWriter(f'runs/AdaDropTB/{model_name}')

        with open(f"{model_name}.log", "w") as log_file:
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, trainloader, optimizer, epoch, log_file, writer)
                test(model, device, testloader, log_file, epoch, writer)
                scheduler.step()

            if args.save_model:
                model_path = f"{args.dataset}_{model_name}.pt"
                torch.save(model.state_dict(), model_path)

        writer.close()


if __name__ == "__main__":
    main()
