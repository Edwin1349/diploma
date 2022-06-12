import cv2
import numpy as np

class Box():
    def __init__(self, x_min, y_min, box_width, box_height, type):
        self.x_min = x_min
        self.y_min = y_min
        self.box_width = box_width
        self.box_height = box_height
        self.type = type
        self.ts_class = "..."
        self.confidence = 0

    def info(self):
        print("type: ", type)

    def set_class(self, class_name, confidence):
        self.ts_class = class_name
        self.confidence = confidence

class YOLO():
    def __init__(self, cfg_path, weights_path, probability_minimum = 0.5, threshold = 0.3):
        self.probability_minimum = probability_minimum
        self.threshold = threshold

        self.detector = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.layers_names_all = self.detector.getLayerNames()
        self.layers_names_output = np.array([self.layers_names_all[i - 1] for i in self.detector.getUnconnectedOutLayers()])

        self.W = None
        self.H = None
        self.bounding_boxes = None
        self.confidences = None
        self.class_numbers = None


    def detect(self, frame):
        (self.H, self.W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.detector.setInput(blob)
        output_from_network = self.detector.forward(self.layers_names_output)

        self.bounding_boxes = []
        self.confidences = []
        self.class_numbers = []

        return self.calculate_boxes(output_from_network)

    def calculate_boxes(self, output_from_network):
        for result in output_from_network:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]

                if confidence_current > self.probability_minimum:
                    box_current = detected_objects[0:4] * np.array([self.W, self.H, self.W, self.H])

                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    self.bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    self.confidences.append(float(confidence_current))
                    self.class_numbers.append(class_current)

        results = cv2.dnn.NMSBoxes(self.bounding_boxes, self.confidences,
                                   self.probability_minimum, self.threshold)
        return results
