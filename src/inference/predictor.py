import cv2
import numpy as np
import mlflow.tensorflow

class Predictor:
    def __init__(self, model_uri, img_size):
        self.model = mlflow.tensorflow.load_model(model_uri)
        self.img_size = tuple(img_size)

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        return np.expand_dims(img, axis=0)

    def predict(self, image_path):
        x = self.preprocess(image_path)
        prob = self.model.predict(x)[0][0]

        return {
            "label": "dog" if prob > 0.5 else "cat",
            "confidence": float(prob if prob > 0.5 else 1 - prob)
        }