import torch
import numpy as np
import torchvision.transforms as transforms
import copy
import os
import pandas as pd
import json

from utils.yolo_tools import (ticks_to_numeric, find_closest_ticks, get_ip_data,
                              get_categorical_output, fill_non_complete_prediction, get_numerical_output)

from utils.util_funcs import (is_numeric, find_duplicate_indices, lowercase_except_first_letter, sort_torch_by_col,
                              graph_class2type, save_bentech_res)
from models.eval_prediction import benetech_score
from models.mga_classifier import GraphClassifier, resize_and_pad
from models.mga_detector import GraphDetecor
from data.global_data import indx_2_chart_label



class MGAPredictor:
    def __init__(self, model, acc_device="cpu", ocr_mode="trocr", classifier_method="classifier",
                 classifier_path=r"..\weights\classifier\resnet_34_999.pth"):
        self.yolo_model = GraphDetecor(model, acc_device, ocr_mode)
        self.classifier_method = classifier_method
        if classifier_method == "classifier":
            model = GraphClassifier(num_classes=5, resnet_model=34)
            model.load_state_dict(torch.load(classifier_path))
            self.classifier = model


    @staticmethod
    def get_output_order(x_output):
        if is_numeric(x_output):
            return np.argsort([float(x) for x in x_output])
        return np.argsort(x_output)

    @staticmethod
    def postprocess_cat(finsl_res, graph_type, horizontal=False):
        x_y, labels, values = ticks_to_numeric(finsl_res, ["y_ticks"] if not horizontal else ["x_ticks"])
        closest_ticks, x_ticks_xy, y_ticks_xy = find_closest_ticks(x_y, labels, graph_type, 1 if not horizontal else 2,
                                                 1 if graph_type == 'dot_point' or horizontal else 2)
        interest_points, x_values, y_values, y_coords, x_coords = get_ip_data(x_y, labels, values, graph_type, True)
        y_output, x_output = get_categorical_output(interest_points, closest_ticks, x_values, y_values, y_coords,
                                                    None if not horizontal else x_coords)
        if graph_type == "line_point": # line points tend to miss intrest points but x values are robust
            x_output_fill, y_output_fill = fill_non_complete_prediction(x_output, x_values, y_output, method="median")
            x_output, y_output = np.append(x_output, x_output_fill), np.append(y_output, y_output_fill)
            out_order = np.argsort(x_ticks_xy[:, 1])
            x_output, y_output = x_output[out_order], y_output[out_order]
        # Drop duplicates if exsits
        dups = find_duplicate_indices(x_output)
        if dups and len(x_output) > len(x_values):
            valid_indices = np.array([i for i in range(len(x_output)) if i not in dups])
            x_output, y_output = x_output[valid_indices], y_output[valid_indices]
        # output_order = MGAPredictor.get_output_order(x_output)
        if not horizontal:
            return lowercase_except_first_letter(x_output), y_output
        else:
            out_order = np.argsort(interest_points[:, 1])
            return x_output[out_order], lowercase_except_first_letter(y_output)[out_order]

    @staticmethod
    def postprocess_numeric(finsl_res):
        x_y, labels, values = ticks_to_numeric(finsl_res, ["x_ticks", "y_ticks"])
        closest_ticks, x_ticks_xy, y_ticks_xy = find_closest_ticks(x_y, labels, "scatter_point", 2, 2)
        interest_points, x_values, y_values, y_coords, x_coords = get_ip_data(x_y, labels, values, "scatter_point", True)
        y_output, x_output = get_numerical_output(interest_points, closest_ticks, x_values, y_values, x_coords,
                                                  y_coords)
        return x_output, y_output

    @staticmethod
    def postprocess_line(finsl_res):
        # TODO special case: shared origin
        return MGAPredictor.postprocess_cat(finsl_res, graph_type="line_point")

    @staticmethod
    def postprocess_dot(finsl_res):
        x_output, y_output = MGAPredictor.postprocess_cat(finsl_res, graph_type="dot_point")
        if is_numeric(x_output):
            return np.array([float(val) for val in x_output]), y_output
        return x_output, y_output

    @staticmethod
    def postprocess_bar(finsl_res, graph_class):
        # TODO check if bar is horizontal or vertical and handle data appropriately
        # TODO special case: histograms
        x_output, y_output = MGAPredictor.postprocess_cat(finsl_res, graph_type="bar",
                                                          horizontal= graph_class == "horizontal_bar")
        return x_output, y_output

    @staticmethod
    def postprocess_scat(finsl_res):
        return MGAPredictor.postprocess_numeric(finsl_res)

    def get_graph_type_class(self, finsl_res, img=None):
        if self.classifier_method == "yolo":
            return self.yolo_model.get_graph_type(finsl_res, img)
        elif self.classifier_method == "classifier":
            img_p = resize_and_pad(transforms.Grayscale(num_output_channels=1)(img)) #preprocess
            graph_class = self.classifier(img_p)
            graph_class = indx_2_chart_label[graph_class]
            ["line", "scatter", "vertical_bar", "horizontal_bar", "dot"]
            return graph_class, graph_class2type(graph_class)

    @staticmethod
    def final_output_to_df(final_output):
        return pd.DataFrame(final_output).set_index("id")

    @staticmethod
    def anot_to_gt(anot, img_name):
        chart_type = anot['chart-type']
        x_data = [dic["x"] for dic in anot['data-series']]
        y_data = [dic["y"] for dic in anot['data-series']]
        if chart_type == "dot":
            if is_numeric(x_data):
                x_data = [float(val) for val in x_data]
        if chart_type == "scatter":
            order = np.argsort(x_data)
            x_data = np.array(x_data)[order].tolist()
            y_data = np.array(y_data)[order].tolist()
        # order = MGAPredictor.get_output_order(x_data)
        # x_data = np.array(x_data)[order]
        # y_data = np.array(y_data)[order]
        gt_output = [{"id": f"{img_name}_x", "data_series": x_data, "chart_type": chart_type},
                     {"id": f"{img_name}_y", "data_series": y_data, "chart_type": chart_type}]
        df_gt = MGAPredictor.final_output_to_df(gt_output)
        return df_gt

    def postprocess(self, finsl_res, img_name, img):
        graph_type, graph_class = self.get_graph_type_class(finsl_res, img)
        if graph_type == "line_point":
            x_output, y_output = self.postprocess_line(finsl_res)
        if graph_type == "dot_point":
            x_output, y_output = self.postprocess_dot(finsl_res)
        if graph_type == "bar":
            x_output, y_output = self.postprocess_bar(finsl_res, graph_class)
        if graph_type == "scatter_point":
            x_output, y_output = self.postprocess_scat(finsl_res)
        final_output = [{"id": f"{img_name}_x", "data_series": x_output, "chart_type": graph_class},
                        {"id": f"{img_name}_y", "data_series": y_output, "chart_type": graph_class}]
        return final_output

    def predict(self, img_list):
        img_names = [os.path.basename(img).split(".")[0] for img in img_list]
        finsl_res, imgs = self.yolo_model.predict(img_list)
        finsl_res_out = list(map(self.postprocess, finsl_res, img_names, imgs))
        return finsl_res_out, img_names

    def get_bentech_score(self, img_path, annot_folder):
        finsl_res_out, img_names = self.predict([img_path])
        df = self.final_output_to_df(finsl_res_out[0])
        annot_path = os.path.join(annot_folder, f"{os.path.basename(img_path).split('.')[0]}.json")
        with open(annot_path) as json_file:
            anot = json.load(json_file)
        df_gt = self.anot_to_gt(anot, img_names[0])
        return finsl_res_out, benetech_score(df_gt, df), df, df_gt


if __name__ == "__main__":
    yolo_path = r"C:\Users\Nir\PycharmProjects\mga\weights\detector\img640_batch_32_lr1e4.pt"
    acc_device = "cuda" if torch.cuda.is_available() else "cpu"
    ocr_mode = "trocr"
    annot_folder = r"D:\train\annotations"
    res_foldr = r"D:\MGA\img_res"
    imgs_dir = r"D:\train\images"
    yolo_model = MGAPredictor(yolo_path, acc_device, ocr_mode)
    imgs_paths = [
        r"D:\MGA\sorted_images\horizontal_bar\1bd01b1c40d8.jpg",
        r"D:\MGA\sorted_images\vertical_bar\0ace695dea6b.jpg",
        r"D:\MGA\sorted_images\vertical_bar\029867ef63f2.jpg",
        # r"D:\MGA\sorted_images\dot\00ae5cc822f0.jpg",
        # r"D:\MGA\sorted_images\vertical_bar\0ad4f865ce03.jpg",
        # r"D:\MGA\sorted_images\dot\00b1597b6970.jpg",
        r"D:\MGA\sorted_images\line\0a225ebe30f6.jpg",
        r"D:\MGA\sorted_images\line\02a23ca8f04b.jpg",
        r"D:\MGA\sorted_images\line\029eb96f26d9.jpg",
        r"D:\MGA\sorted_images\scatter\029d1a7e17d2.jpg",
        # r"D:\MGA\sorted_images\scatter\00ade5dc4f7a.jpg",
        # r"D:\MGA\sorted_images\scatter\1a3d883bf388.jpg"
                  ]
    imgs_paths = imgs_paths + [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)]
    for img_path in imgs_paths:
        try:
            finsl_res_out, benetech_score_eval, df_out, df_gt = yolo_model.get_bentech_score(img_path, annot_folder)
            save_bentech_res(img_path, res_foldr, benetech_score_eval, df_out, df_gt)
        except Exception as e:
            print(e)
            save_bentech_res(img_path, res_foldr, benetech_score_eval, failed=True)
