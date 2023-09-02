import os

import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils.util_funcs import sort_torch_by_col
from PIL import Image
from math import floor, ceil
# import pytesseract
import easyocr
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import matplotlib.pyplot as plt


def preprocess_image(image):
    """Convert the image to grayscale and threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = (image.shape[1] // 10, image.shape[0] // 10)
    kernel_size = (kernel_size[0] + 1 if kernel_size[0] % 2 == 0 else kernel_size[0],
                   kernel_size[1] + 1 if kernel_size[1] % 2 == 0 else kernel_size[1])
    kernel = np.ones(kernel_size, np.uint8)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

    return angle


def detect_rotation(image):
    thresh = preprocess_image(image)
    return get_rotation_angle(thresh)


def straighten_image(image, angle):
    """Straighten the image based on the given angle without cutting off any part."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute the new dimensions of the image after rotation
    alpha = np.deg2rad(angle)  # angle in radians
    new_width = int(abs(h * np.sin(alpha)) + abs(w * np.cos(alpha)))
    new_height = int(abs(h * np.cos(alpha)) + abs(w * np.sin(alpha)))

    # Adjust the rotation matrix to take into account the translation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_width - w) / 2
    M[1, 2] += (new_height - h) / 2

    rotated = cv2.warpAffine(image, M, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated


def extract_middle_contour(image, thresh):
    """Extract the central part of the image based on contours."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Assuming the largest contour corresponds to the central part of the image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    img_h = image.shape[0]
    cropped = image[max(y-img_h//10, 0):min(y + h+img_h//10, img_h), :]

    return cropped


def rotate_and_crop(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    angle = detect_rotation(image)
    angle_45 = 35 < abs(angle) < 55
    angle_135 = 125 < abs(angle) < 155
    if angle_45 or angle_135:
        if angle_45:
            angle = 45
        if angle_135:
            angle = 135
        image = straighten_image(image, angle-90)
        thresh = preprocess_image(image)
        image = extract_middle_contour(image, thresh)
    plt.imshow(image)
    plt.show()
    return image


def compute_roi_coordinates(x, y, w, h):
    """
    Compute the coordinates of the region of interest.

    Parameters:
    - x, y: Center coordinates of the bounding box.
    - w, h: Width and height of the bounding box.

    Returns:
    - Coordinates (x1, y1, x2, y2) of the region of interest.
    """
    x1 = int(floor(x - w / 2))
    y1 = int(floor(y - h / 2))
    x2 = int(ceil(x + w / 2))
    y2 = int(ceil(y + h / 2))
    return x1, y1, x2, y2


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
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def initialize_ocr_resources(mode, gpu=False):
    """
    Initialize and return required resources based on the OCR mode.

    Parameters:
    - mode: OCR algorithm to use. Options: easyocr, trocr.
    - gpu: Use GPU for computations (applicable for EasyOCR).

    Returns:
    - Tuple containing the initialized resources. (reader for EasyOCR, processor and model for TrOCR)
    """
    reader, processor, model = None, None, None

    if mode == "easyocr":
        reader = easyocr.Reader(['en'], gpu=gpu)
    elif mode == "trocr":
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
        model.config.max_length = 50
        if gpu:
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return reader, processor, model


def extract_text_from_boxes(img, boxes, mode="tesseract", gpu=False, reader=None, processor=None, model=None):
    """
    Extract text from given bounding boxes in an image.

    Parameters:
    - img: Input image as a numpy array.
    - boxes: List of bounding boxes. Each box is a tuple (x_center, y_center, width, height).
    - mode: OCR algorithm to use. Options: easyocr, trocr, tesseract.

    Returns:
    - List of strings extracted from each bounding box.
    """
    texts, rois = [], []
    img_pil = Image.fromarray(img)

    # Initialize required resources based on mode
    if all(x is None for x in (reader, processor, model)) and mode != "tesseract":
        reader, processor, model = initialize_ocr_resources(mode, gpu=gpu)

    for box in boxes:
        x1, y1, x2, y2 = compute_roi_coordinates(*box)
        roi = img_pil.crop((x1, y1, x2, y2))
        roi = rotate_and_crop(roi)
        if mode == "easyocr":
            texts.append(extract_with_easyocr(roi, reader))
        elif mode == "trocr":
            rois.append(roi)
        else:
            texts.append(extract_with_tesseract(roi))
    if mode == "trocr":
        texts = extract_with_trocr(rois, processor, model)
    return texts


def tick_label2axis_label(box_torch):
    tick_labels = box_torch[box_torch[:, 4] == 7]
    plot_xywh = box_torch[box_torch[:, 4] == 0][0]
    min_x, max_y = plot_xywh[0] - plot_xywh[2] / 2, plot_xywh[1] + plot_xywh[3] / 2
    x_tick_labels = sort_torch_by_col(tick_labels[tick_labels[:, 1] > max_y], 0)
    y_tick_labels = sort_torch_by_col(tick_labels[tick_labels[:, 0] < min_x], 1)
    return x_tick_labels, y_tick_labels


def tick_label2value(box_torch, img):
    # TODO take tick label bbox and extract the relevant text, turn it to numerical / categorical
    # both x,y tick labels are categorical we have an issue one should be numerical, generally the y axis is numerical
    # unless we are in horizontal bar mode
    pass


def match_tick2label(box_torch):
    x_tick_labels, y_tick_labels = tick_label2axis_label(box_torch)
    x_ticks_location = sort_torch_by_col(box_torch[box_torch[:, 4] == 1], 0)
    y_ticks_location = sort_torch_by_col(box_torch[box_torch[:, 4] == 2], 1)
    # now ticks and labels are in the same order so x_tick 1 matches x_label 1
    # TODO tick_label2value


def validate_length():
    # TODO validate the length of x_ticks and intrest points (line_point,bar, dot)
    pass
