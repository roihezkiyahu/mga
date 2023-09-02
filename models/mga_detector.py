from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from data.global_data import idx_to_class_box
from utils.util_funcs import sort_torch_by_col
from utils.yolo_tools import tick_label2axis_label, extract_text_from_boxes, initialize_ocr_resources


class GraphDetecor:
    def __init__(self, model, acc_device="cpu", ocr_mode="trocr"):
        # TODO add cuda support
        if isinstance(model, str):
            self.model = YOLO(model)
        else:
            self.model = model
        self.ocr_mode = ocr_mode
        self.acc_device = acc_device
        self.ocr_models = initialize_ocr_resources(ocr_mode, gpu=False)

    def predict(self, img_list):
        res = self.model(img_list, augment=False, iou=0.5)
        boxes = [torch.cat((result.boxes.xywh, result.boxes.cls.unsqueeze(1)), dim=1) for result in res]
        imgs = [result.orig_img for result in res]
        finsl_res = [self.reformat_boxes(box_torch, img, img_res) for box_torch, img, img_res in zip(boxes, imgs, res)]
        # box_classes = ["plot", "x_tick", "y_tick", "scatter_point", "bar", "dot_point", "line_point", "tick_label"]

    def reformat_boxes(self, box_torch, img, res):
        plt.imshow(res.plot(font_size=0.5, labels=False))
        plt.show()
        box_torch = sort_torch_by_col(sort_torch_by_col(box_torch, 0), 4)
        box_torch_no_label = box_torch[~torch.isin(box_torch[:, 4], torch.tensor([1, 2, 7]))]

        x_tick_labels, y_tick_labels = tick_label2axis_label(box_torch)
        x_extracted_text = extract_text_from_boxes(img, x_tick_labels[:, :4], self.ocr_mode, self.acc_device =="cuda",
                                                   *self.ocr_models)
        print(x_extracted_text)
        y_extracted_text = extract_text_from_boxes(img, y_tick_labels[:, :4], self.ocr_mode, self.acc_device =="cuda",
                                                   *self.ocr_models)
        print(y_extracted_text)

        final_x_y_data = torch.cat([box_torch_no_label[:, :2], x_tick_labels[:, :2], y_tick_labels[:, :2]])
        d_type = ["plot_bb"] + [idx_to_class_box[label.item()] for label in box_torch_no_label[1:,4]]+ \
                 ["x_tick"]*len(x_tick_labels) + ["y_tick"]*len(y_tick_labels)
        values = [float('nan')]*len(box_torch_no_label) + x_extracted_text + y_extracted_text
        return final_x_y_data, d_type, values


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