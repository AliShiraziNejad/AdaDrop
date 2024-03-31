import torch.nn as nn
import torch.nn.functional as F

from AdaDrop import AdaDrop

import einops


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

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=128, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

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

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=128, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(p=0.25)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')

        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.dropout(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

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

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=128, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.adaDrop1 = AdaDrop(self.fc1, scaling="inverse", N=1)
        self.adaDrop2 = AdaDrop(self.fc2, scaling="inverse", N=1)
        self.adaDrop3 = AdaDrop(self.fc3, scaling="inverse", N=1)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')

        x = self.AdaDrop(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.AdaDrop(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.AdaDrop(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

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

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=128, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.adaDrop1 = AdaDrop(self.fc1, scaling="softmax", N=1)
        self.adaDrop2 = AdaDrop(self.fc2, scaling="softmax", N=1)
        self.adaDrop3 = AdaDrop(self.fc3, scaling="softmax", N=1)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')

        x = self.AdaDrop1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.AdaDrop2(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.AdaDrop3(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

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

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_features=128, out_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.adaDrop1 = AdaDrop(self.fc1, scaling="norm", N=1)
        self.adaDrop2 = AdaDrop(self.fc2, scaling="norm", N=1)
        self.adaDrop3 = AdaDrop(self.fc3, scaling="norm", N=1)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')

        x = self.AdaDrop1(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.AdaDrop2(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.AdaDrop3(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

        return output
