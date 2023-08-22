import ast
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import json
import cv2
from data.global_data import class_box_to_idx, outlier_images
import random
import shutil


def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except:
        return value


def create_dataframe(annotations_directory):
    """
    creates a dataframe form annotations
    """
    files = os.listdir(annotations_directory)
    json_files = [file for file in files if file.endswith('.json')]
    data_list = []
    images_dir = os.path.join(os.path.dirname(annotations_directory), "images")
    for json_file in tqdm(json_files, total=len(json_files), desc='Processing JSON files'):
        with open(os.path.join(annotations_directory, json_file)) as f:
            data = pd.json_normalize(json.load(f))
            name = json_file.replace('.json', '')
            data['name'] = name
            data['annotation'] = os.path.join(annotations_directory, json_file)
            data['image'] = os.path.join(images_dir, f"{name}.jpg")
            data_list.append(data)
    df = pd.concat(data_list, ignore_index=True)
    return df


def get_bboxes(row, width=12):
    boxes = []
    # Plot bounding box
    boxes.append({
        "class": "plot",
        "x": row["plot-bb.x0"] + row["plot-bb.width"] / 2,
        "y": row["plot-bb.y0"] + row["plot-bb.height"] / 2,
        "width": row["plot-bb.width"],
        "height": row["plot-bb.height"]
    })
    # X-axis ticks - int
    for x_tick in safe_literal_eval(row["axes.x-axis.ticks"]):
        tick_pt = x_tick['tick_pt']
        boxes.append({
            "class": "x_tick",
            "x": tick_pt['x'],
            "y": tick_pt['y'],
            "width": width,
            "height": width
        })
    # Y-axis ticks - int
    for y_tick in safe_literal_eval(row["axes.y-axis.ticks"]):
        tick_pt = y_tick['tick_pt']
        boxes.append({
            "class": "y_tick",
            "x": tick_pt['x'],
            "y": tick_pt['y'],
            "width": width,
            "height": width
        })
    # Scatter points - float
    scatter_points = safe_literal_eval(row["visual-elements.scatter points"])
    if scatter_points:
        if len(scatter_points) > 1:
            for point in scatter_points:
                boxes.append({
                    "class": "scatter_point",
                    "x": point["x"],
                    "y": point["y"],
                    "width": width,
                    "height": width
                })
        else:
            for point in scatter_points[0]:
                boxes.append({
                    "class": "scatter_point",
                    "x": point["x"],
                    "y": point["y"],
                    "width": width,
                    "height": width
                })
    # Bars - float
    for bar in safe_literal_eval(row["visual-elements.bars"]):
        if "x0" in bar.keys():
            boxes.append({
                "class": "bar",
                "x": bar["x0"] + bar["width"] / 2,
                "y": bar["y0"],
                "width": width,
                "height": width
            })
        else:
            boxes.append({
                "class": "bar",
                "x": bar["x"],
                "y": bar["y"],
                "width": width,
                "height": width
            })
    # Dot points - float
    # This assumes that row["visual-elements.dot points"] is a DataFrame
    elements_dots = safe_literal_eval(row["visual-elements.dot points"])
    if elements_dots:
        if len(elements_dots) > 1:
            res_frame = pd.DataFrame(elements_dots)
        else:
            res_frame = pd.DataFrame(elements_dots[0])
        last_x_min = 0
        while True:
            x_min = res_frame["x"][last_x_min < res_frame["x"]].min()
            valid_range = res_frame[(last_x_min < res_frame["x"]) & (res_frame["x"] < x_min + 10)]
            x = valid_range["x"].mean()
            y = valid_range["y"].min()
            last_x_min = valid_range["x"].max()
            if np.isnan(x):
                break
            dot_size = valid_range["y"].max() - (row["plot-bb.y0"] + row["plot-bb.height"])
            boxes.append({
                "class": "dot_point",
                "x": x,
                "y": y,
                "width": width,
                "height": width
            })

    elements_lines = safe_literal_eval(row["visual-elements.lines"])
    if elements_lines:
        if len(elements_lines) > 1:
            for line in elements_lines:
                boxes.append({
                    "class": "line_point",
                    "x": line["x"],
                    "y": line["y"],
                    "width": width,
                    "height": width
                })
        else:
            for line in elements_lines[0]:
                boxes.append({
                    "class": "line_point",
                    "x": line["x"],
                    "y": line["y"],
                    "width": width,
                    "height": width
                })

    elements_text = safe_literal_eval(row["text"])
    if elements_text:
        for polygon in elements_text:
            if polygon["role"] == "tick_label":
                x_0 = min(polygon["polygon"]['x0'], polygon["polygon"]['x3'])
                x_1 = max(polygon["polygon"]['x1'], polygon["polygon"]['x2'])
                y_0 = min(polygon["polygon"]['y0'], polygon["polygon"]['y1'])
                y_1 = max(polygon["polygon"]['y2'], polygon["polygon"]['y3'])
                boxes.append({
                    "class": "tick_label",
                    "x": (x_0 + x_1) / 2,
                    "y": (y_0 + y_1) / 2,
                    "width": x_1 - x_0,
                    "height": y_1 - y_0
                })
    return boxes


def create_bounding_boxes(df, width=12):
    bounding_boxes = []
    for i, row in tqdm(df.iterrows()):
        bounding_boxes.append(get_bboxes(row, width))
    return bounding_boxes


def annotation_to_labels(image_path, chart_boxes, is_box=True, labels_folder="labels"):
    if not is_box:
        chart_boxes = get_bboxes(chart_boxes, width=12)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    file_name = os.path.basename(image_path)
    os.makedirs(labels_folder, exist_ok=True)

    with open(os.path.join(labels_folder, file_name.split(".")[0] + '.txt'), 'w') as f:
        for box in chart_boxes:
            box["x"] = (box["x"]) / width
            box["y"] = (box["y"]) / height
            if box["class"] in ["plot", "tick_label"]:
                box["width"] = box["width"] / width
                box["height"] = box["height"] / height
            else:
                box["width"] = 0.02  # box["width"]/width
                box["height"] = 0.02 * width / height  # box["height"]/height
            box["class"] = class_box_to_idx[box["class"]]
            f.write(f"{box['class']} {box['x']} {box['y']} {box['width']} {box['height']}\n")


def load_bounding_boxes(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    bounding_boxes = []
    for line in lines:
        values = line.strip().split()
        box = {
            'class': int(values[0]),
            'x': float(values[1]),
            'y': float(values[2]),
            'width': float(values[3]),
            'height': float(values[4])
        }
        bounding_boxes.append(box)

    return bounding_boxes


def copy_files(file_list, source_img, source_lbl, dest_img, dest_lbl):
    for f in tqdm(file_list):
        if f.split(".")[0] in outlier_images:
            continue
        if os.path.exists(os.path.join(source_img, f)) and os.path.exists(
                os.path.join(source_lbl, f.replace('.jpg', '.txt'))):
            shutil.copy(os.path.join(source_img, f), os.path.join(dest_img, f))
            shutil.copy(os.path.join(source_lbl, f.replace('.jpg', '.txt')),
                        os.path.join(dest_lbl, f.replace('.jpg', '.txt')))
        else:
            print("no both for file: ", f)


def sort_yolo_folders(tr_img, tr_labels):
    random.seed(42)
    base_dir = 'dataset'
    subfolders = ['train', 'valid', 'test']
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_dir, subfolder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, subfolder, 'labels'), exist_ok=True)
    all_images = [f for f in os.listdir(tr_img) if f.endswith('.jpg')]

    # Shuffle and split the data
    random.shuffle(all_images)
    train_split = int(0.80 * len(all_images))
    valid_split = int(0.15 * len(all_images)) + train_split

    train_images = all_images[:train_split]
    valid_images = all_images[train_split:valid_split]
    test_images = all_images[valid_split:]

    # Copy files to the new directory structure
    copy_files(train_images, tr_img, tr_labels, os.path.join(base_dir, 'train', 'images'),
               os.path.join(base_dir, 'train', 'labels'))
    copy_files(valid_images, tr_img, tr_labels, os.path.join(base_dir, 'valid', 'images'),
               os.path.join(base_dir, 'valid', 'labels'))
    copy_files(test_images, tr_img, tr_labels, os.path.join(base_dir, 'test', 'images'),
               os.path.join(base_dir, 'test', 'labels'))


def get_random_image(dir_path):
    images = [os.path.join(dir_path, "images", f) for f in os.listdir(os.path.join(dir_path, "images")) if f.endswith('.jpg')]
    return random.choice(images)
