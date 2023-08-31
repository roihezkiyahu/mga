import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from PIL import Image, ImageChops
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch import nn
from pytorch_lightning import Trainer
from torchvision.transforms.functional import pad
from torch.utils.data import random_split
# from lightning.pytorch import loggers as pl_loggers
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
try:
    from data.global_data import indx_2_chart_label, chart_labels_2_indx
except ImportError:
    chart_labels = ["line", "scatter", "vertical_bar", "horizontal_bar", "dot"]
    chart_labels_2_indx = {class_name: idx for idx, class_name in enumerate(chart_labels)}
    indx_2_chart_label = {idx: class_name for idx, class_name in enumerate(chart_labels)}
from plot_functions.mga_plt import plot_misclassified_images


class ChartDataLoader(Dataset):
    def __init__(self, train_df, labels=[], transform=None, with_name=False):
        self.train_df = train_df
        if len(labels):
            self.labels_2_indx = dict(zip(labels, range(len(labels))))
            self.indx_2_label = dict(zip(range(len(labels)), labels))
        else:
            self.labels_2_indx = chart_labels_2_indx
            self.indx_2_label = indx_2_chart_label
        self.transform = transform
        self.with_name = with_name

    def __getitem__(self, index):
        image_path = self.train_df["image"][index]
        chart_type = self.train_df["chart-type"][index]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        chart = self.labels_2_indx[chart_type]
        if self.with_name:
            return image, chart, os.path.basename(image_path)
        return image, chart

    def __len__(self):
        return len(self.train_df)


class GraphClassifier(pl.LightningModule):
    def __init__(self, num_classes=5, optimizer='Adam', lr=0.001, one_channel=True,
                 class_weights=[1, 2.0, 1.3, 5.0, 3.0]):
        super().__init__()
        base_model = resnet50(weights=None)
        if one_channel:
            base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = base_model.fc.in_features
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Remove original FC layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        weights = torch.tensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        self.criterion = criterion
        self.optimizer_type = optimizer
        self.lr = lr

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten features
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_type == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_type == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Optimizer type {self.optimizer_type} not recognized.')
        scheduler = StepLR(optimizer, step_size=3, gamma=0.9)
        return [optimizer], [scheduler]


def resize_and_pad(img, size):
    # Get original aspect ratio
    aspect_ratio = img.size[0] / img.size[1]

    if aspect_ratio > 1: # width > height
        new_width = size
        new_height = int(size / aspect_ratio)
    else: # height > width
        new_height = size
        new_width = int(size * aspect_ratio)

    # Resize the image
    img = img.resize((new_width, new_height))

    # Calculate padding
    pad_left = (size - new_width) // 2
    pad_right = size - new_width - pad_left
    pad_top = (size - new_height) // 2
    pad_bottom = size - new_height - pad_top

    # Pad the image
    img = pad(img, (pad_left, pad_top, pad_right, pad_bottom))

    return img


def invert_img(img, prob=0.25):
    if np.random.rand() < prob:
        inverted_img = ImageChops.invert(img)
        return inverted_img
    else:
        return img


def get_transforms(image_size,
                   num_erases=10,
                   erase_params=None,
                   grayscale_channels=1,
                   flip_prob=0.5,
                   affine_translate=(0.2, 0.2),
                   affine_scale=(0.8, 1.2),
                   use_color_jitter=False,
                   use_invert_img=False,
                   use_erasing_transforms=False,
                   gray_scale=True):
    if erase_params is None:
        erase_params = [
            {'p': 0.9, 'scale': (0.0005, 0.005), 'ratio': (0.33, 3), 'value': 0, 'inplace': False},
            {'p': 0.9, 'scale': (0.0005, 0.005), 'ratio': (0.33, 3), 'value': 1, 'inplace': False}
        ]

    erasing_transforms = [transforms.RandomErasing(**param) for param in erase_params for _ in range(num_erases)]

    transform_list_train = [
        transforms.RandomHorizontalFlip(flip_prob),
        transforms.Lambda(lambda img: resize_and_pad(img, size=image_size)),
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=0, translate=affine_translate, scale=affine_scale)
    ]

    if gray_scale:
        transform_list_train.insert(0, transforms.Grayscale(num_output_channels=grayscale_channels))

    if use_color_jitter:
        transform_list_train.insert(1, transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))

    if use_invert_img:
        transform_list_train.insert(2, transforms.Lambda(lambda img: invert_img(img)))

    if use_erasing_transforms:
        transform_list_train.extend(erasing_transforms)

    transform_train = transforms.Compose(transform_list_train)

    transform_list_val = [
        transforms.Lambda(lambda img: resize_and_pad(img, size=image_size)),
        transforms.ToTensor(),
    ]

    if gray_scale:
        transform_list_val.insert(0, transforms.Grayscale(num_output_channels=grayscale_channels))

    transform_val = transforms.Compose(transform_list_val)

    return transform_train, transform_val


def evaluate_model(model, val_dataloader, acc_device, plot_misclassified=True, with_name=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            if with_name:
                inputs, labels, names = batch
            else:
                inputs, labels = batch
                names = []
            inputs = inputs.to(acc_device)
            labels = labels.to(acc_device)

            outputs = model.to(acc_device)(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if plot_misclassified:
                plot_misclassified_images(inputs, labels, preds, names)

    return all_preds, all_labels


if __name__ == "__main__":
    # Parmas
    batch_size = 16
    validation_ratio = 0.2
    image_size = 224
    learning_rate = 5e-5
    model = GraphClassifier(num_classes=5, lr=learning_rate)
    model.load_state_dict(torch.load('model_weights.pth'))


