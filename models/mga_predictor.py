import torch
import numpy as np
import torchvision.transforms as transforms
import copy
import os
import pandas as pd
import json
import cv2
import re
from PIL import Image

from data.global_data import class_box_to_idx, outlier_images
from utils.yolo_tools import (ticks_to_numeric, find_closest_ticks, get_ip_data,
                              get_categorical_output, fill_non_complete_prediction, get_numerical_output)

from utils.util_funcs import (is_numeric, find_duplicate_indices, lowercase_except_first_letter, sort_torch_by_col,
                              graph_class2type, save_bentech_res, replace_infinite)
from models.eval_prediction import benetech_score
from models.mga_classifier import GraphClassifier, resize_and_pad
from models.mga_detector import GraphDetecor
from data.global_data import indx_2_chart_label


class MGAPredictor:
    def __init__(self, model, acc_device="cpu", ocr_mode="trocr", classifier_method="comb",
                 classifier_path=r"..\weights\classifier\resnet_34_999.pth", ocr_model_paths={},
                 iou=0.5, conf=0.05):
        self.yolo_model = GraphDetecor(model, acc_device, ocr_mode, ocr_model_paths=ocr_model_paths,
                                       iou=iou, conf=conf)
        self.classifier_method = classifier_method
        if classifier_method in ["classifier", "comb"]:
            model = GraphClassifier(num_classes=5, resnet_model=34)
            model.load_state_dict(torch.load(classifier_path))
            self.classifier = model


    @staticmethod
    def get_output_order(x_output):
        if is_numeric(x_output):
            return np.argsort([float(x) for x in x_output])
        return np.argsort(x_output)

    @staticmethod
    def drop_dups(x_output, y_output, x_values, y_values, horizontal=False):
        if not horizontal:
            dups = find_duplicate_indices(x_output)
            if dups and len(x_output) > len(x_values):
                valid_indices = np.array([i for i in range(len(x_output)) if i not in dups])
                x_output, y_output = x_output[valid_indices], y_output[valid_indices]
            return x_output, y_output

        dups = find_duplicate_indices(y_output)
        if dups and len(y_output) > len(y_values):
            valid_indices = np.array([i for i in range(len(y_output)) if i not in dups])
            x_output, y_output = x_output[valid_indices], y_output[valid_indices]
        return x_output, y_output


    @staticmethod
    def fill_output_missing_val(graph_type, x_output, y_output, x_values, y_values, horizontal=False):
        x_output_fill = []
        if graph_type in ["line_point", 'dot_point']: # line points tend to miss intrest points but x values are robust
            method = "median" if graph_type == "line_point" else 0
            x_output_fill, y_output_fill = fill_non_complete_prediction(x_output, x_values, y_output, method=method)
            x_output, y_output = np.append(x_output, x_output_fill), np.append(y_output, y_output_fill)
        if horizontal: # line points tend to miss intrest points but x values are robust
            y_output_fill, x_output_fill = fill_non_complete_prediction(y_output, y_values, x_output, method="median")
            x_output, y_output = np.append(x_output, x_output_fill), np.append(y_output, y_output_fill)
        filled = len(x_output_fill) > 0
        return x_output, y_output, filled

    @staticmethod
    def validate_xy_values(x_values, y_values, cat=True):
        if not len(y_values) + len(x_values):
            if cat:
                return False, ["abc", "def"], [0, 1]
            else:
                return False, [0, 1], [0, 1]
        if not len(y_values):
            return False, x_values, [0]*len(x_values)
        if not len(x_values):
            if cat:
                return False, ["abc"]*len(y_values), y_values
            else:
                return False, [0] * len(y_values), y_values
        return True, [], []

    @staticmethod
    def postprocess_cat(finsl_res, graph_type, horizontal=False):
        x_y, labels, values = ticks_to_numeric(finsl_res, ["y_ticks"] if not horizontal else ["x_ticks"])
        closest_ticks, x_ticks_xy, y_ticks_xy = find_closest_ticks(x_y, labels, graph_type, 1 if not horizontal else 2,
                                                 1 if graph_type == 'dot_point' or horizontal else 2)
        interest_points, x_values, y_values, y_coords, x_coords, logic_vec = get_ip_data(x_y, labels, values,
                                                                                         graph_type, True, horizontal)
        if horizontal:
            y_order = interest_points[:,1].argsort()
            interest_points, closest_ticks = interest_points[y_order, :], [closest_ticks[place] for place in y_order]
        interest_points, closest_ticks = interest_points[logic_vec], [c_tick for c_tick, val in zip(closest_ticks,
                                                                                                    logic_vec) if val]
        valid_xy, x_output, y_output = MGAPredictor.validate_xy_values(x_values, y_values)
        if not valid_xy:
            return x_output, y_output
        y_output, x_output = get_categorical_output(interest_points, closest_ticks, x_values, y_values, y_coords,
                                                    None if not horizontal else x_coords)
        # Drop duplicates if exsits
        x_output, y_output = MGAPredictor.drop_dups(x_output, y_output, x_values, y_values, horizontal)

        x_output, y_output, filled = MGAPredictor.fill_output_missing_val(graph_type, x_output, y_output,
                                                                          x_values, y_values, horizontal)
        if graph_type in ["line_point", 'dot_point'] and filled:
            out_order = [np.where(x_output == x)[0][0] for x in x_values if x in x_output]
            x_output, y_output = x_output[out_order], y_output[out_order]
        if horizontal and filled:
            out_order = [np.where(y_output == y)[0][0] for y in y_values if y in y_output]
            x_output, y_output = x_output[out_order], y_output[out_order]
        # output_order = MGAPredictor.get_output_order(x_output)
        return x_output, y_output


    @staticmethod
    def postprocess_numeric(finsl_res):
        x_y, labels, values = ticks_to_numeric(finsl_res, ["x_ticks", "y_ticks"])
        closest_ticks, x_ticks_xy, y_ticks_xy = find_closest_ticks(x_y, labels, "scatter_point", 2, 2)
        interest_points, x_values, y_values, y_coords, x_coords, logic_vec = get_ip_data(x_y, labels, values,
                                                                                         "scatter_point", True)
        interest_points, closest_ticks = interest_points[logic_vec], [c_tick for c_tick, val in zip(closest_ticks,
                                                                                                    logic_vec) if val]
        valid_xy, x_output, y_output = MGAPredictor.validate_xy_values(x_values, y_values, False)
        if not valid_xy:
            return x_output, y_output
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
        if is_numeric(x_output, dot=True):
            x_output = [re.sub(r'[a-zA-Z]', '', val) for val in x_output]
            return np.array([float(val) if val not in ["None_ocr_val", ""] else 0 for val in x_output]), y_output
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
        if self.classifier_method == "comb":
            graph_type, graph_class = self.yolo_model.get_graph_type(finsl_res, img)
            if graph_type not in ["bar", "plot_bb"]: # yolo doew not detect well horizontal vs vertical graphs
                return graph_type, graph_class
        if self.classifier_method == "yolo":
            graph_type, graph_class = self.yolo_model.get_graph_type(finsl_res, img)
            if graph_type == "plot_bb":
                graph_type = "line_point"
                graph_class = "line"
            return graph_type, graph_class
        if self.classifier_method in ["classifier", "comb"]:
            img_pil = Image.fromarray(img)
            grayscale_transform = transforms.Grayscale(num_output_channels=1)
            img_p = resize_and_pad(grayscale_transform(img_pil))
            img_p = transforms.ToTensor()(img_p)
            self.classifier.eval()
            graph_class = self.classifier(img_p.unsqueeze(0))
            graph_class = indx_2_chart_label[torch.argmax(graph_class).item()]
            return graph_class2type(graph_class), graph_class

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
        if isinstance(x_data[0], float) or isinstance(x_data[0], int):
            x_data = list(np.array(x_data)[~np.isnan(x_data)])
        if isinstance(y_data[0], float) or isinstance(y_data[0], int):
            y_data = list(np.array(y_data)[~np.isnan(y_data)])
        gt_output = [{"id": f"{img_name}_x", "data_series": x_data, "chart_type": chart_type},
                     {"id": f"{img_name}_y", "data_series": y_data, "chart_type": chart_type}]
        df_gt = MGAPredictor.final_output_to_df(gt_output)
        return df_gt

    def postprocess(self, finsl_res, img_name, img):
        graph_type, graph_class = self.get_graph_type_class(finsl_res, img)
        print(graph_type, graph_class)
        if graph_type == "line_point":
            x_output, y_output = self.postprocess_line(finsl_res)
            y_output = replace_infinite(y_output)
        if graph_type == "dot_point":
            x_output, y_output = self.postprocess_dot(finsl_res)
            y_output = replace_infinite(y_output)
        if graph_type == "bar":
            x_output, y_output = self.postprocess_bar(finsl_res, graph_class)
            y_output = replace_infinite(y_output)
            x_output = replace_infinite(x_output)
        if graph_type == "scatter_point":
            x_output, y_output = self.postprocess_scat(finsl_res)
            y_output = replace_infinite(y_output)
            x_output = replace_infinite(x_output)

        if isinstance(x_output[0], str):
            x_output = [x if x != "None_ocr_val" else "" for x in x_output]
        if isinstance(y_output[0], str):
            y_output = [y if y != "None_ocr_val" else "" for y in y_output]

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
        print(df.iloc[0,0])
        print(df.iloc[1,0])
        print(df_gt.iloc[0, 0])
        print(df_gt.iloc[1, 0])
        return finsl_res_out, benetech_score(df_gt, df), df, df_gt


if __name__ == "__main__":
    yolo_path = r"C:\Users\Nir\PycharmProjects\mga\weights\detector\img640_batch_32_lr1e4_iou5_bg6000_aug60k_cont_long.pt"
    acc_device = "cuda" if torch.cuda.is_available() else "cpu"
    ocr_mode = "paddleocr"
    annot_folder = r"D:\train\annotations"
    res_foldr = r"G:\My Drive\MGA\img_res_aug_detector_extracted_run26" #r"D:\MGA\img_res"
    imgs_dir = "D:\MGA\sorted_images\extracted"
    # imgs_dir = r"G:\My Drive\MGA\zero_score_exctracted"
    non_imgs_zero = [file.split("_")[0] for file in os.listdir(r"G:\My Drive\MGA\img_res_aug_detector_extracted") if
                 not (file.endswith("_0.csv") or file.endswith("_gt.csv"))]
    yolo_model = MGAPredictor(yolo_path, acc_device, ocr_mode, iou=0.5)
    overwrite = True
    save_res = True
    imgs_paths_0 = [
        r"D:\MGA\sorted_images\dot\00b1597b6970.jpg"
        # r"D:\train\images\00d76c715deb.jpg",
        # r"D:\train\images\0073ac9cd239.jpg",
        #
        # r"D:\train\images\007a18eb4e09.jpg",
        # r"D:\train\images\00d76c715deb.jpg",
        # r"D:\train\images\00f5404753cf.jpg",
        #
        # r"D:\train\images\0116c35ada8c.jpg",
        # r"D:\train\images\01850b694f00.jpg",

        # r"D:\MGA\sorted_images\line\00a68f5c2a93.jpg",
        # r"D:\MGA\sorted_images\line\02a23ca8f04b.jpg",
        # r"D:\MGA\sorted_images\line\029eb96f26d9.jpg",
        # r"D:\MGA\sorted_images\scatter\029d1a7e17d2.jpg",
        # r"D:\MGA\sorted_images\horizontal_bar\1bd01b1c40d8.jpg",
        # r"D:\MGA\sorted_images\vertical_bar\0ace695dea6b.jpg",
        # r"D:\MGA\sorted_images\vertical_bar\029867ef63f2.jpg",
        # r"D:\MGA\sorted_images\dot\00ae5cc822f0.jpg",
        # r"D:\MGA\sorted_images\vertical_bar\0ad4f865ce03.jpg",
        # r"D:\MGA\sorted_images\dot\00b1597b6970.jpg",
        # r"D:\MGA\sorted_images\scatter\00ade5dc4f7a.jpg",
        # r"D:\MGA\sorted_images\scatter\1a3d883bf388.jpg"
                  ]

    res_files = [file.split("_")[0] for file in os.listdir(res_foldr)]
    imgs_paths = [os.path.join(imgs_dir, img) for img in os.listdir(imgs_dir)
                                 if img.split(".")[0] not in outlier_images + non_imgs_zero and img.endswith(".jpg") ]
    for img_path in imgs_paths_0 + imgs_paths[148:]:
        try:
            img_name = os.path.basename(img_path).split(".")[0]
            if not overwrite:
                if img_name in res_files and not np.any([img_name in img_p for img_p in imgs_paths_0]):
                    print(img_name, " already processes")
                    continue
            print(img_name)
            finsl_res_out, benetech_score_eval, df_out, df_gt = yolo_model.get_bentech_score(img_path, annot_folder)
            save_bentech_res(img_path, res_foldr, benetech_score_eval, df_out, df_gt, save_res=save_res)
        except Exception as e:
            print(e)
            save_bentech_res(img_path, res_foldr, "", failed=True, save_res=save_res)
