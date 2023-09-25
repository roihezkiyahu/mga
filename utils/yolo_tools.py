import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils.util_funcs import sort_torch_by_col, remove_characters
from PIL import Image
from math import floor, ceil
# import pytesseract
import easyocr
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
from utils.util_funcs import lowercase_except_first_letter


def preprocess_image(image):
    """Convert the image to grayscale and threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = (int(image.shape[0] / 7.5), int(image.shape[0] / 7.5))
    kernel_size = (kernel_size[0] + 1 if kernel_size[0] % 2 == 0 else kernel_size[0],
                   kernel_size[1] + 1 if kernel_size[1] % 2 == 0 else kernel_size[1])
    kernel = np.ones(kernel_size, np.uint8)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(thresh) > 125: # of of the image is dark (dark mode, gray, etc)
        thresh = 255 - thresh
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated


def get_rotation_angle(thresh):
    """Detect rotation angle from the thresholded image based on the largest contour."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box_sorted = sorted(box, key=lambda x: x[0])
    left_tilt = box_sorted[0][1] < box_sorted[-1][1]
    angle = rect[-1]
    if left_tilt:
        angle += 90
    if thresh.shape[0] > thresh.shape[1]*1.5:
        angle = 90

    return angle


def detect_rotation(image):
    thresh = preprocess_image(image)
    return get_rotation_angle(thresh)


def straighten_image(image, angle):
    """Straighten the image based on the given angle without cutting off any part."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute the new dimensions of the image after rotation
    alpha = np.deg2rad(angle)
    new_width = int(abs(h * np.sin(alpha)) + abs(w * np.cos(alpha)))
    new_height = int(abs(h * np.cos(alpha)) + abs(w * np.sin(alpha)))

    # Adjust the rotation matrix to take into account the translation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_width - w) / 2
    M[1, 2] += (new_height - h) / 2
    median_blue = np.median(image[:, :, 0])
    median_green = np.median(image[:, :, 1])
    median_red = np.median(image[:, :, 2])
    rotated = cv2.warpAffine(image, M, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(median_red, median_green, median_blue))

    return rotated


def extract_middle_contour(image, thresh):
    """Extract the central part of the image based on contours."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    img_h = image.shape[0]
    cropped = image[max(y-img_h//10, 0):min(y + h+img_h//10, img_h), :]

    return cropped


def rotate_and_crop(image, x_label):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    angle = detect_rotation(image)
    angle_45 = 35 < abs(angle) < 55
    angle_135 = 125 < abs(angle) < 155
    angle_90 = 80 < abs(angle) < 100
    if angle_45 or angle_135 or angle_90:
        if angle_45:
            angle = 45
        if angle_135:
            angle = 135
        if angle_90:
            if x_label:
                angle = 360
            else:
                angle = 180
        image = straighten_image(image, angle-90)
        thresh = preprocess_image(image)
        image = extract_middle_contour(image, thresh)
    return image, angle_45, angle_135


def compute_roi_coordinates(x, y, w, h, offset=2):
    """
    Compute the coordinates of the region of interest.

    Parameters:
    - x, y: Center coordinates of the bounding box.
    - w, h: Width and height of the bounding box.

    Returns:
    - Coordinates (x1, y1, x2, y2) of the region of interest.
    """
    x1 = int(floor(x - w / 2)) - offset
    y1 = int(floor(y - h / 2)) - offset
    x2 = int(ceil(x + w / 2)) + offset
    y2 = int(ceil(y + h / 2)) + offset
    return x1, y1, x2, y2


def get_dims(rois):
    max_width = max(roi.shape[1] for roi in rois)
    max_height = max(roi.shape[0] for roi in rois)
    return int(max_width * 1.1), int(max_height * 1.1)


def get_padding(roi, max_width, max_height):
    pad_height = max_height - roi.shape[0]
    pad_width = max_width - roi.shape[1]
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    return pad_top, pad_bottom, pad_left, pad_right


def add_border(roi):
    return np.pad(roi, ((5, 5), (5, 5), (0, 0)), 'constant', constant_values=0)


def pad_roi(roi, max_width, max_height):
    pad_top, pad_bottom, pad_left, pad_right = get_padding(roi, max_width, max_height)
    return np.pad(roi, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant', constant_values=255)


def get_side_length(padded_rois):
    total_area = sum([roi.shape[0] * roi.shape[1] for roi in padded_rois])
    return int(np.sqrt(total_area))


def get_num_cols_rows(padded_rois, max_width, side_length):
    num_cols = side_length // max_width
    num_rows = int(np.ceil(len(padded_rois) / num_cols))
    return num_cols, num_rows


def create_blank_image(max_width, max_height):
    return np.ones((max_height, max_width, 3), dtype=np.uint8) * 255


def create_composite_image(padded_rois, num_cols):
    num_blank_images = (num_cols - (len(padded_rois) % num_cols)) % num_cols
    max_height, max_width = padded_rois[0].shape[:2]
    padded_rois.extend([create_blank_image(max_width, max_height) for _ in range(num_blank_images)])

    rows = [np.hstack(padded_rois[i:i + num_cols]) for i in range(0, len(padded_rois), num_cols)]
    return np.vstack(rows)


def preprocess_paddle_ocr(rois):
    max_width, max_height = get_dims(rois)
    padded_rois = [add_border(pad_roi(roi, max_width, max_height)) for roi in rois]
    side_length = get_side_length(padded_rois)
    num_cols, num_rows = get_num_cols_rows(padded_rois, max_width, side_length)
    return create_composite_image(padded_rois, num_cols)


def extract_with_easyocr(roi, reader):
    """
    Extract text using EasyOCR.

    Parameters:
    - roi: Region of interest (cropped image section).
    - reader: EasyOCR Reader instance.

    Returns:
    - Extracted text.
    """
    result = reader.readtext(np.array(roi))
    return result[0][1].strip() if result else ""


def extract_with_tesseract(roi):
    """
    Extract text using pytesseract.

    Parameters:
    - roi: Region of interest (cropped image section).

    Returns:
    - Extracted text.
    """
    return pytesseract.image_to_string(roi).strip()


def extract_with_trocr(roi, processor, model):
    """
    Extract text using TrOCR.

    Parameters:
    - roi: Region of interest (cropped image section).
    - processor: TrOCR Processor instance.
    - model: TrOCR VisionEncoderDecoderModel instance.

    Returns:
    - Extracted text.
    """
    pixel_values = processor(images=roi, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    lower_res = lowercase_except_first_letter(processor.batch_decode(generated_ids, skip_special_tokens=True))
    return lower_res


def extract_with_paddleocr(rois, paddleocr):
    paddleocr_res = [paddleocr.ocr(roi, det=False) for roi in rois]
    return [res[0][0][0] if res[0][0][0] != "" else "None_ocr_val" for res in paddleocr_res]


def initialize_ocr_resources(mode, gpu=False, model_paths={}):
    """
    Initialize and return required resources based on the OCR mode.
    """
    reader, processor, model, paddleocr = None, None, None, None

    if mode == "easyocr":
        reader = easyocr.Reader(['en'], gpu=gpu)
    elif mode in ["trocr", "paddleocr"]:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
        model.config.max_length = 50
        if gpu:
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if mode == "paddleocr":
        if model_paths:
            paddleocr = PaddleOCR(use_angle_cls=True, lang='en',
                      det_model_dir=model_paths["det_model_dir"],
                      rec_model_dir=model_paths["rec_model_dir"],
                      cls_model_dir=model_paths["cls_model_dir"])
        else:
            paddleocr = PaddleOCR(use_angle_cls=True, lang='en')
    return reader, processor, model, paddleocr


def extract_text_from_boxes(img, boxes, mode="tesseract", gpu=False, reader=None, processor=None, model=None,
                            paddleocr=None, x_label=False):
    """
    Extract text from given bounding boxes in an image.

    Parameters:
    - img: Input image as a numpy array.
    - boxes: List of bounding boxes. Each box is a tuple (x_center, y_center, width, height).
    - mode: OCR algorithm to use. Options: easyocr, trocr, tesseract.

    Returns:
    - List of strings extracted from each bounding box.
    """
    if not len(boxes):
        return [], False, False
    texts, rois = [], []
    img_pil = Image.fromarray(img)
    rotated_45_list, rotated_135_list = [], []
    if all(x is None for x in (reader, processor, model)) and mode != "tesseract":
        reader, processor, model, paddleocr = initialize_ocr_resources(mode, gpu=gpu)
    for box in boxes:
        x1, y1, x2, y2 = compute_roi_coordinates(*box)
        y1 += int((y2-y1)*0.1)
        roi = img_pil.crop((x1, y1, x2, y2))
        roi, rotated_45, rotated_135 = rotate_and_crop(roi, x_label)
        rotated_45_list.append(rotated_45)
        rotated_135_list.append(rotated_135)
        if mode == "easyocr":
            texts.append(extract_with_easyocr(roi, reader))
        elif mode in ["trocr", "paddleocr"]:
            rois.append(roi)
        else:
            texts.append(extract_with_tesseract(roi))
    if mode == "trocr":
        texts = extract_with_trocr(rois, processor, model)
    if mode == "paddleocr":
        texts = extract_with_paddleocr(rois, paddleocr)
        # if "None_ocr_val" in texts:
        #     mask = [text == "None" for text in texts]
        #     texts[mask] = extract_with_trocr(rois[mask], processor, model)
    return texts, np.mean(rotated_45_list) > 0.5, np.mean(rotated_135_list) > 0.5


def tick_label2axis_label(box_torch):
    tick_labels = box_torch[box_torch[:, 4] == 7]
    if torch.sum(box_torch[:, 4] == 0) > 0:
        plot_xywh = box_torch[box_torch[:, 4] == 0][0]
        min_x, max_y = plot_xywh[0] - plot_xywh[2] / 2, plot_xywh[1] + plot_xywh[3] / 2
    else:
        min_x, max_y = np.inf, 0
    x_tick_labels = sort_torch_by_col(tick_labels[tick_labels[:, 1] > max_y], 0)
    y_tick_labels = sort_torch_by_col(tick_labels[tick_labels[:, 0] < min_x], 1)
    return x_tick_labels, y_tick_labels


def point_to_numeric(val, chars_to_remove):
    try:
        return float(remove_characters(val, chars_to_remove))
    except ValueError:
        return np.inf


def ticks_to_numeric(finsl_res, tick_to_turn=["y_ticks"], chars_to_remove=["%", "$", ",", "C"]):
    x_y, labels, values, _ = finsl_res
    values = np.array(values, dtype=object)
    if "y_ticks" in tick_to_turn:
        mask = np.array(labels) == "y_tick"
        values[mask] = [point_to_numeric(val, chars_to_remove) for val in values[mask]]
    mask = np.array(labels) == "x_tick"
    if "x_ticks" in tick_to_turn:
        values[mask] = [point_to_numeric(val, chars_to_remove) for val in values[mask]]
    # output_order = np.argsort(x_y[mask][:, 0])
    inf_mask = [np.isinf(val) if isinstance(val, float) else False for val in values]
    x_y[inf_mask] = torch.tensor([np.inf, np.inf])
    return x_y, labels, values


def find_closest_ticks(tensor, labels, label_to_find, n_x_ticks, n_y_ticks):
    label_to_find_mask = np.array(labels) == label_to_find
    x_tick_mask = np.array(labels) == 'x_tick'
    y_tick_mask = np.array(labels) == 'y_tick'

    target_points = tensor[label_to_find_mask]
    x_ticks = tensor[x_tick_mask]
    y_ticks = tensor[y_tick_mask]

    closest_ticks = []
    for target_point in target_points:
        distances_to_x_ticks = np.abs(x_ticks[:, 0] - target_point[0])
        distances_to_y_ticks = np.abs(y_ticks[:, 1] - target_point[1])

        closest_x_ticks = distances_to_x_ticks.argsort()[:n_x_ticks]
        closest_y_ticks = distances_to_y_ticks.argsort()[:n_y_ticks]

        closest_ticks.append((closest_x_ticks, closest_y_ticks))

    return closest_ticks, x_ticks, y_ticks


def get_ip_data(x_y, labels, values, label_to_find, include_x_coors=False):
    intrest_points = x_y[np.array([label == label_to_find for label in labels])]
    x_values_mask = np.array([label == "x_tick" for label in labels])
    y_tick_mask = np.array([label == "y_tick" for label in labels])
    x_values = values[x_values_mask]
    y_values = values[y_tick_mask]
    y_coords = x_y[y_tick_mask]
    if include_x_coors:
        return intrest_points, x_values, y_values, y_coords, x_y[x_values_mask]
    return intrest_points, x_values, y_values, y_coords


def get_categorical_output(interest_points, closest_ticks, x_values, y_values, y_coords, x_coords=None):
    y_output, x_output = [], []
    for interest_point, close_tick in zip(interest_points, closest_ticks):
        x_loc, y_loc = close_tick
        x_tick, y_vals = x_values[x_loc], y_values[y_loc]
        if len(x_loc) == 1 and len(y_loc) == 1:
            y_output.append(y_vals)
            x_output.append(x_tick)
            continue
        if isinstance(x_coords, type(None)):
            xy_1, xy_2 = y_coords[y_loc]
            pixel_val = (y_vals[0] - y_vals[1]) / (xy_2[1] - xy_1[1])
            interest_point_y_val = pixel_val * (xy_2[1] - interest_point[1]) + y_vals[1]

            y_output.append(interest_point_y_val.item())
            x_output.append(x_tick)
        else:
            x_xy_1, x_xy_2 = x_coords[x_loc]
            x_vals = x_values[x_loc]
            pixel_val = (x_vals[1] - x_vals[0]) / (x_xy_2[0] - x_xy_1[0])
            interest_point_x_val = pixel_val * (interest_point[0] - x_xy_1[0]) + x_vals[0]
            y_output.append(y_vals)
            x_output.append(interest_point_x_val.item())

    return np.array(y_output), np.array(x_output)


def get_numerical_output(interest_points, closest_ticks, x_values, y_values, x_coords, y_coords):
    y_output, x_output = [], []
    for interest_point, close_tick in zip(interest_points, closest_ticks):
        x_loc, y_loc = close_tick
        x_tick, y_vals, x_vals = x_values[x_loc], y_values[y_loc], x_values[x_loc]
        y_xy_1, y_xy_2 = y_coords[y_loc]
        x_xy_1, x_xy_2 = x_coords[x_loc]

        pixel_val = (y_vals[0] - y_vals[1]) / (y_xy_2[1] - y_xy_1[1])
        interest_point_y_val = pixel_val * (y_xy_2[1] - interest_point[1]) + y_vals[1]

        pixel_val = (x_vals[1] - x_vals[0]) / (x_xy_2[0] - x_xy_1[0])
        interest_point_x_val = pixel_val * (interest_point[0] - x_xy_1[0]) + x_vals[0]

        y_output.append(interest_point_y_val.item())
        x_output.append(interest_point_x_val.item())

    return np.array(y_output), np.array(x_output)


def fill_non_complete_prediction(x_output, x_values, y_output, method="median"):
    # better to give wrong prediction than to give short prediction
    not_in_x_output = [value for value in x_values if value not in x_output]
    # #might want to try some smarter prediciton based on close points
    # not_in_x_output_index = np.where(np.isin(x_values, not_in_x_output))[0]
    # not_in_x_output_xy = x_ticks_xy[not_in_x_output_index]
    if isinstance(method, int):
        new_y_out = [method] * len(not_in_x_output)
        return not_in_x_output, new_y_out
    if method == "median":
        med_val = np.median(y_output)
        new_y_out = [med_val] * len(not_in_x_output)
        return not_in_x_output, new_y_out
    if method == "nan":
        new_y_out = [np.nan] * len(not_in_x_output)
        return not_in_x_output, new_y_out
    mean_val = np.median(y_output)
    new_y_out = [mean_val] * len(not_in_x_output)
    return not_in_x_output, new_y_out


def classify_bars(image, bbox):
    try:
        if image is None:
            return "vertical_bar"
        h, w = image.shape[:2]
        image = image[max(int(bbox[1]-bbox[3]/2), 0): min(int(bbox[1]+bbox[3]/2), h),
                      max(int(bbox[0] - bbox[2] / 2), 0): min(int(bbox[0] + bbox[2] / 2), w)]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 200, 250)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        horizontal_count = 0
        vertical_count = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > h:
                horizontal_count += 1
            else:
                vertical_count += 1

        if vertical_count >= horizontal_count:
            return "vertical_bar"
        else:
            return "horizontal_bar"

    except Exception as e:
        print(e)
        return "vertical_bar"


def keep_one_class(folder_path, class_2_keep='6'):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), 'r') as f:
                    lines = f.readlines()

                with open(os.path.join(root, file), 'w') as f:
                    for line in lines:
                        if line.startswith(class_2_keep):
                            f.write(line)
