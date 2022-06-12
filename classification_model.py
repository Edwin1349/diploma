import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2

classes = np.array(["speed limit 20", "speed limit 30", "speed limit 50", "speed limit 60", "speed limit 70", "speed limit 80",
         "restriction ends 80", "speed limit 100", "speed limit 120", "no overtaking", "no overtaking (trucks)",
         "priority at next intersection", "priority road", "give way", "stop", "no traffic both ways", "no trucks",
         "no entry", "danger", "bend left", "bend right", "bend", "uneven road", "slippery road", "road narrows",
         "construction", "traffic signal", "pedestrian crossing", "school crossing", "cycles crossing", "snow", "animals",
         "restriction ends", "go right", "go left", "go straight", "go right or straight", "go left or straight",
         "keep right", "keep left", "roundabout", "restriction ends", "restriction ends (overtaking (trucks))"])


types = {0: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16], # Prohibitory category
         1: [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], # Danger category
         2: [33, 34, 35, 36, 37, 38, 39, 40], # Mandatory category
         3: [6, 12, 13, 14, 17, 32, 41, 42]} # Other category

class TrafficSignNet(nn.Module):
    def __init__(self, output_dim=len(classes)):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 1000),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x

train_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((112, 112)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
val_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((112, 112)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

inv_transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])