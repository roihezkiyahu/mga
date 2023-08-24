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
from data.global_data import box_classes, colors_list

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