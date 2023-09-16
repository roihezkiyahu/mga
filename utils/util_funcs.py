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
from typing import Union, List, Iterable


def extract_width_and_height(df):
    """Extracts the width and height of each image in the dataset.

    Args:
        df (Pandas DataFrame): The Pandas DataFrame containing the annotations.

    Returns:
        widths (list): The list of widths of the images.
        heights (list): The list of heights of the images.

    """
    image_paths = df.image.values
    widths = np.array([])
    heights = np.array([])
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        width, height = image.shape[:2]
        widths = np.append(widths, width)
        heights = np.append(heights, height)
    return widths, heights


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


def get_bboxes(row, width=12, gen=False, only_plot_area=False):
    boxes = []
    # Plot bounding box
    boxes.append({
        "class": "plot",
        "x": row["plot-bb.x0"] + row["plot-bb.width"] / 2,
        "y": row["plot-bb.y0"] + row["plot-bb.height"] / 2,
        "width": row["plot-bb.width"],
        "height": row["plot-bb.height"]
    })
    if only_plot_area:
        return boxes
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
        if len(scatter_points) > 1 and gen:
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
                "x": bar["x0"] + bar["width"] / 2 if row["chart-type"] == "vertical_bar" else bar["x0"] + bar["width"],
                "y": bar["y0"] if row["chart-type"] == "vertical_bar" else bar["y0"] + bar["height"] / 2,
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
        if len(elements_dots) > 1 and gen:
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
            if len(valid_range) > 1:
                dot_size = np.nanmedian(np.diff(valid_range["y"].sort_values()))
            else:
                dot_size = ((row["plot-bb.y0"] + row["plot-bb.height"]) - valid_range["y"].max())*2
            boxes.append({
                "class": "dot_point",
                "x": x,
                "y": y-dot_size/2,
                "width": width,
                "height": width
            })

    elements_lines = safe_literal_eval(row["visual-elements.lines"])
    if elements_lines:
        if len(elements_lines) > 1 and gen:
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


def annotation_to_labels(image_path, chart_boxes, is_box=True, labels_folder="labels", img_folder="", gen=False,
                         only_plot_area=False):
    if not is_box:
        # TODO make sure it works correctly when reading data from files and not from flow
        chart_boxes = get_bboxes(chart_boxes, width=12, gen=gen, only_plot_area=only_plot_area)
    image = cv2.imread(image_path)
    if isinstance(image, type(None)):
        print(image_path, " is None")
        return
    height, width = image.shape[:2]
    file_name = os.path.basename(image_path)
    os.makedirs(labels_folder, exist_ok=True)
    if not is_box:
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
    else:
        shutil.copy(chart_boxes, os.path.join(labels_folder, file_name.split(".")[0] + '.txt'))
    if img_folder != "":
        cv2.imwrite(os.path.join(img_folder, file_name), image)


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


def copy_files(file_list, source_img, source_lbl, dest_img, dest_lbl, valid_names=[], overwrite=False):
    for f in tqdm(file_list):
        f_name = f.split(".")[0]

        if f_name in outlier_images:
            continue

        if valid_names and f_name not in valid_names:
            continue

        src_img_path = os.path.join(source_img, f)
        src_lbl_path = os.path.join(source_lbl, f.replace('.jpg', '.txt'))
        dest_img_path = os.path.join(dest_img, f)
        dest_lbl_path = os.path.join(dest_lbl, f.replace('.jpg', '.txt'))

        if not os.path.exists(src_img_path) or not os.path.exists(src_lbl_path):
            print("No both for file:", f)
            continue

        if os.path.exists(src_img_path) and (overwrite or not os.path.exists(dest_img_path)):
            shutil.copy(src_img_path, dest_img_path)

        if os.path.exists(src_lbl_path) and (overwrite or not os.path.exists(dest_lbl_path)):
            shutil.copy(src_lbl_path, dest_lbl_path)


def sort_yolo_folders(tr_img, tr_labels, valid_names=[], overwrite=False, base_dir='dataset'):
    random.seed(42)
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
               os.path.join(base_dir, 'train', 'labels'), valid_names, overwrite=overwrite)
    copy_files(valid_images, tr_img, tr_labels, os.path.join(base_dir, 'valid', 'images'),
               os.path.join(base_dir, 'valid', 'labels'), valid_names, overwrite=overwrite)
    copy_files(test_images, tr_img, tr_labels, os.path.join(base_dir, 'test', 'images'),
               os.path.join(base_dir, 'test', 'labels'), valid_names, overwrite=overwrite)


def get_random_image(dir_path):
    images = [os.path.join(dir_path, "images", f) for f in os.listdir(os.path.join(dir_path, "images")) if f.endswith('.jpg')]
    return random.choice(images)


def get_gen_label(img_name):
    img_name = img_name.split(".")[0]
    if "line" in img_name:
        return "line"
    if "scatter" in img_name or "scat" in img_name:
        return "scatter"
    if "vertical" in img_name:
        return "vertical_bar"
    if "horizontal" in img_name:
        return "horizontal_bar"
    if "dot" in img_name:
        return "dot"


def create_gen_df(gen_folder):
    imgs_list_paths = [os.path.join(gen_folder, image) for image in os.listdir(gen_folder) if image.endswith(".jpg")]
    chart_types = [get_gen_label(image) for image in os.listdir(gen_folder) if image.endswith(".jpg")]
    return pd.DataFrame({"image": imgs_list_paths,
                 "chart-type": chart_types})


def sort_torch_by_col(torch_input, col=1):
    return torch_input[torch_input[:, col].argsort()]


def is_numeric(value):
    def check_numeric(single_value):
        try:
            float(single_value)
            return True
        except ValueError:
            return False

    if isinstance(value, Iterable):
        return all(check_numeric(single_value) for single_value in value)
    else:
        return check_numeric(value)


def remove_characters(s, chars_to_remove):
    for char in chars_to_remove:
        s = s.replace(char, "")
    return s


def find_duplicate_indices(lst):
    value_indices = {}
    duplicate_indices = []

    for index, item in enumerate(lst):
        if item in value_indices:
            duplicate_indices.append(index)
        else:
            value_indices[item] = index

    return duplicate_indices


def lowercase_except_first_letter(arr):
    if arr.ndim != 1:
        return "Input should be a 1-dimensional numpy array"

    def modify_string(s):
        if len(s) > 1:
            return s[0] + s[1:].lower()
        else:
            return s

    vectorized_modify_string = np.vectorize(modify_string)
    return vectorized_modify_string(arr)

def graph_class2type(graph_class):
    if graph_class == "line":
        graph_type = "line_point"
    if graph_class == "scatter":
        graph_type = "scatter_point"
    if graph_class == "bar":
        graph_type = "line_point"
    if graph_class == "bar":
        graph_type = "line_point"
    if graph_class == "dot":
        graph_type = "dot_point"
    return graph_type


def save_bentech_res(img_path, res_foldr, benetech_score_eval, df_out=pd.DataFrame({}), df_gt=pd.DataFrame({}),
                     failed=False):
    img_name = os.path.basename(img_path).split(".")[0]
    if not failed:
        df_name = os.path.join(res_foldr, f"{img_name}_{int(benetech_score_eval * 100)}.csv")
        df_gt_name = os.path.join(res_foldr, f"{img_name}_gt.csv")
        print(benetech_score_eval)
    else:
        img_name = os.path.basename(img_path).split(".")[0]
        df_name = os.path.join(res_foldr, f"{img_name}_fail.csv")
        df_gt_name = os.path.join(res_foldr, f"{img_name}_gt.csv")
        print("failed:", img_path)
    df_out.to_csv(df_name)
    df_gt.to_csv(df_gt_name)




if __name__ == "__main__":
    labels_folder = r"C:\Users\Nir\Downloads\labels"
    label_files = os.listdir(labels_folder)
    column_names = ["class_id", "x_center", "y_center", "width", "height"]
    for file in tqdm(label_files):
        file_path = os.path.join(labels_folder, file)
        df = pd.read_csv(file_path, sep=" ", header=None, names=column_names)
        if not len(df):
            print("no labels: ", file_path)
    print("Done")

