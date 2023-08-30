import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import random
from sklearn.linear_model import LinearRegression
import numpy as np
import json
import os
import re
from PIL import Image, ImageChops
import cv2
import matplotlib.patches as patches
import pandas as pd
from IPython.display import display
from tqdm import tqdm
import random
from utils.util_funcs import load_bounding_boxes
from data.global_data import box_classes, colors_list, chart_labels
try:
    from data.global_data import indx_2_chart_label, chart_labels_2_indx
except ImportError:
    chart_labels_2_indx = {class_name: idx for idx, class_name in enumerate(chart_labels)}
    indx_2_chart_label = {idx: class_name for idx, class_name in enumerate(chart_labels)}
from sklearn.metrics import confusion_matrix
import seaborn as sns




class_2_color = dict(zip(box_classes, colors_list))

color_mapping = {
    "r": (255, 0, 0),
    "blue": (0, 0, 255),
    "black": (0, 0, 0),
    "purple": (128, 0, 128),
    "yellow": (255, 255, 0),
    "green": (0, 128, 0),
    "orange": (255, 165, 0)
}

class_2_color_cv = {box_class: color_mapping[color] for box_class, color in zip(box_classes, colors_list)}


def plot_image_with_boxes(img_path, boxes, jupyter=True):
    img = cv2.imread(img_path)

    # Iterate through the boxes and draw rectangles
    for box in boxes:
        top_left = (int(box["x"] - box["width"] / 2), int(box["y"] - box["height"] / 2))
        bottom_right = (int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2))
        color = class_2_color_cv[box['class']]
        color = tuple([int(c * 255) for c in color])  # Convert color to range [0, 255]
        cv2.rectangle(img, top_left, bottom_right, color, 1)

    # Convert BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use PIL to display the image in Jupyter
    if jupyter:
        display(Image.fromarray(img_rgb))
    else:
        plt.imshow(img_rgb)
        plt.show()


def plot_w_box_from_path(index=0, base_path=os.path.join('dataset', 'train')):
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')
    images_list = os.listdir(images_path)

    # Load image
    img_path = os.path.join(images_path, images_list[index])
    image_name = os.path.basename(img_path)
    label_name = image_name.split(".")[0] + ".txt"
    label_path = os.path.join(labels_path, label_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    boxes = load_bounding_boxes(label_path)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(img)

    # Create a Rectangle patch
    for box in boxes:
        point = (box["x"] * img_w - box["width"] * img_w / 2, box["y"] * img_h - box["height"] * img_h / 2)
        rect = patches.Rectangle(
            point,  # Shift to make (x, y) the center
            box["width"] * img_w,
            box["height"] * img_h,
            linewidth=1,
            edgecolor=class_2_color[box_classes[box['class']]],
            facecolor='none'
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def show_images_in_grid(dataloader, n_rows, n_cols):
    # Get a batch of images and labels
    images, labels = next(iter(dataloader))

    # Prepare the matplotlib figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    for i, ax in enumerate(axs.flat):
        # Make sure we don't go over the number of images in the batch
        if i >= len(images):
            break

        # Get the label for the image
        label = indx_2_chart_label[labels[i].item()]

        # Convert image from PyTorch tensor to NumPy array
        img = images[i].permute(1, 2, 0).numpy()

        # Display the image and its label
        if img.shape[2] == 1:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(label)
        ax.axis('off')

    plt.show()


def plot_metrics_from_version(version, base_path="/kaggle/working/logs/csv/lightning_logs/"):
    metrics = pd.read_csv(f"{base_path}version_{version}/metrics.csv")

    train_acc_epoch = metrics["train_acc_epoch"][~metrics["train_acc_epoch"].isna()]
    val_acc_epoch = metrics["val_acc_epoch"][~metrics["val_acc_epoch"].isna()]
    epochs_number = list(range(1, len(train_acc_epoch) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_number, train_acc_epoch, label='Train Accuracy', marker='o')
    plt.plot(epochs_number, val_acc_epoch, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy across Steps')
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.ylim(0.95, 1)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_misclassified_images(inputs, labels, preds):
    wrong_idxs = (preds != labels).nonzero(as_tuple=True)[0]
    for idx in wrong_idxs:
        plt.figure(figsize=(4,4))
        plt.imshow(inputs[idx].cpu().permute(1,2,0).numpy(), cmap="gray")
        plt.title(f"True: {indx_2_chart_label[labels[idx].item()]}, Predicted: {indx_2_chart_label[preds[idx].item()]}")
        plt.axis('off')
        plt.show()


def compute_and_plot_confusion_matrix(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel('Actual labels')
    plt.xlabel('Predicted labels')
    plt.title('Confusion Matrix')
    plt.show()
