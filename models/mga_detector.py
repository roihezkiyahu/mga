from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from plot_functions.mga_plt import plot_image_with_boxes
from data.global_data import idx_to_class_box
from utils.util_funcs import sort_torch_by_col
from utils.yolo_tools import (tick_label2axis_label, extract_text_from_boxes, initialize_ocr_resources,
                              classify_bars, post_process_texts, nms_with_confs)
import copy
import numpy as np
import os

class GraphDetecor:
    def __init__(self, model, acc_device="cpu", ocr_mode="paddleocr", iou=0.5, conf=0.15, show_res=True,
                 ocr_model_paths={}):
        # TODO add cuda support
        if isinstance(model, str):
            self.model = YOLO(model)
        else:
            self.model = model
        self.ocr_mode = ocr_mode
        self.acc_device = acc_device
        self.ocr_models = initialize_ocr_resources(ocr_mode, gpu=acc_device=="cuda" and torch.cuda.is_available(),
                                                   model_paths=ocr_model_paths)
        self.iou = iou
        self.conf = conf
        self.show_res = show_res

    @staticmethod
    def get_graph_type(finsl_res, img=None):
        final_res_copy = copy.deepcopy(finsl_res[1])
        items_to_remove = {"tick_label", "plot", "x_tick", "y_tick"}
        final_res_copy = [item for item in final_res_copy if item not in items_to_remove]
        unique_values, counts = np.unique(final_res_copy, return_counts=True)
        graph_type = unique_values[np.argmax(counts)]
        if graph_type == "line_point":
            graph_class = "line"
        if graph_type == "dot_point":
            graph_class = "dot"
        if graph_type == "bar":
            graph_class = classify_bars(img, finsl_res[3])
        if graph_type == "scatter_point":
            graph_class = "scatter"
        if graph_type == "plot_bb":
            graph_class = "plot_bb"
        return graph_type, graph_class

    def predict(self, img_list):
        res = self.model(img_list, augment=False, iou=self.iou, conf=self.conf)
        boxes = [torch.cat((result.boxes.xywh, result.boxes.cls.unsqueeze(1)), dim=1) for result in res]
        confs = [result.boxes.conf for result in res]
        imgs = [result.orig_img for result in res]
        finsl_res = [self.reformat_boxes(box_torch, img, img_res, conf, img_n) for box_torch, img, img_res, conf, img_n
                     in zip(boxes, imgs, res, confs, img_list)]
        return finsl_res, imgs
        # box_classes = ["plot", "x_tick", "y_tick", "scatter_point", "bar", "dot_point", "line_point", "tick_label"]

    def reformat_boxes(self, box_torch, img, res, confs, img_n):
        if self.show_res:
            plt.imshow(img)
            plt.show()
            plt.imshow(res.plot(font_size=0.5, labels=False))
            plt.show()

        # plt.imshow(res.plot(font_size=0.5, labels=False))
        # plt.savefig(fr"D:\anotated_imgs\{os.path.basename(img_n.split('.')[0])}_w_dets.jpg")
        # plt.close()
        box_torch = nms_with_confs(box_torch, confs, 0.1)
        box_torch = sort_torch_by_col(sort_torch_by_col(box_torch, 0), 4)
        box_torch_no_label = box_torch[~torch.isin(box_torch[:, 4], torch.tensor([1, 2, 7]).to(self.acc_device))]
        if box_torch_no_label[0, 4] == 0:
            condition1 = box_torch_no_label[:, 0] <= box_torch_no_label[0, 0] + box_torch_no_label[0, 2] / 2 + 5
            condition2 = box_torch_no_label[:, 1] >= box_torch_no_label[0, 1] - box_torch_no_label[0, 3] / 2 - 5
            condition3 = box_torch_no_label[:, 1] <= box_torch_no_label[0, 1] + box_torch_no_label[0, 3] / 2 + 5
            condition4 = torch.logical_or(box_torch_no_label[:, 0] >= box_torch_no_label[0, 0] -
                                          box_torch_no_label[0, 2] / 2 - 5, box_torch_no_label[:, 4] != 3)

            box_torch_no_label = box_torch_no_label[torch.logical_and(torch.logical_and(condition1, condition2),
                                                                      torch.logical_and(condition3, condition4))]
        box_torch_no_label = torch.cat([box_torch_no_label[0,:].unsqueeze(0),
                                        sort_torch_by_col(box_torch_no_label[1:,:], 0)])
        x_tick_labels, y_tick_labels = tick_label2axis_label(box_torch)
        y_tick_labels = sort_torch_by_col(y_tick_labels, 1)
        x_extracted_text, rot_45_x, rot_135_x = extract_text_from_boxes(img, x_tick_labels[:, :4], self.ocr_mode,
                                                                    self.acc_device =="cuda", *self.ocr_models,
                                                                        x_label=True)
        y_extracted_text, rot_45_y, rot_135_y = extract_text_from_boxes(img, y_tick_labels[:, :4], self.ocr_mode,
                                                                    self.acc_device =="cuda", *self.ocr_models)

        x_extracted_text = post_process_texts(x_extracted_text)
        y_extracted_text = post_process_texts(y_extracted_text)
        if rot_45_x: # change tick loc if there was a rotation
            x_tick_labels[:, 0] = x_tick_labels[:, 0] + x_tick_labels[:, 2] /2
        if rot_135_x: # change tick loc if there was a rotation
            x_tick_labels[:, 0] = x_tick_labels[:, 0] - x_tick_labels[:, 2] /2
        final_x_y_data = torch.cat([box_torch_no_label[:, :4], x_tick_labels[:, :4], y_tick_labels[:, :4]])
        d_type = [idx_to_class_box[box_torch_no_label[0,4].item()].replace("plot", "plot_bb")]\
                 + [idx_to_class_box[label.item()] for label in box_torch_no_label[1:, 4]] + \
                 ["x_tick"]*len(x_tick_labels) + ["y_tick"]*len(y_tick_labels)
        values = [float('nan')]*len(box_torch_no_label) + x_extracted_text + y_extracted_text
        # if self.show_res:
        #     final_x_y_data_ext = torch.cat([box_torch_no_label[:, :5], x_tick_labels[:, :5], y_tick_labels[:, :5]])
        #     plot_image_with_boxes(img, [dict(zip(["x", "y", "width", "height", "class"], box.astype(int))) for box in
        #                                 final_x_y_data_ext.numpy()], False)
        return final_x_y_data, d_type, values, box_torch[0, :4].tolist()


if __name__ == "__main__":
    yolo_path = r"C:\Users\Nir\PycharmProjects\mga\weights\detector\img640_batch_32_lr1e4.pt"
    acc_device = "cuda" if torch.cuda.is_available() else "cpu"
    ocr_mode = "trocr"
    yolo_model =GraphDetecor(yolo_path, acc_device, ocr_mode)
    imgs_paths = [
                  r"D:\MGA\sorted_images\line\0a225ebe30f6.jpg",
                  r"D:\MGA\sorted_images\line\000ec434b697.jpg",
                  r"D:\MGA\sorted_images\line\0aa4801c493f.jpg",
                  r"D:\MGA\sorted_images\line\0b5b34bc4d0e.jpg",
                  r"D:\MGA\sorted_images\dot\00e555427ef0.jpg",
                  r"D:\MGA\sorted_images\dot\00ea170dc993.jpg",
                  r"D:\MGA\sorted_images\dot\0a3d04fea0ef.jpg",
                  r"D:\MGA\sorted_images\line\000cd10b6cda.jpg",
                  r"D:\MGA\sorted_images\line\00c461cc3116.jpg",
                  r"D:\MGA\sorted_images\dot\00b1597b6970.jpg",
                  r"D:\MGA\sorted_images\dot\00d7276ef6d1.jpg",
                  ]
    yolo_model.predict(imgs_paths)
