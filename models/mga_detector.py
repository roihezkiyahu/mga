from ultralytics import YOLO


class GraphDetecor:
    def __init__(self, model):
        # TODO add cuda support
        if isinstance(model, str):
            self.model = YOLO(model)
        else:
            self.model = model

    def predict(self, img_list):
        res = self.model(img_list)
        boxes = [(result.boxes.xywh[:2], result.boxes.cls) for result in res]
