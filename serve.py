from fastapi import FastAPI, UploadFile
import mlflow.tensorflow
import numpy as np
import cv2

app = FastAPI()

model = mlflow.tensorflow.load_model("models:/cat-dog-classifier/Production")

IMG_SIZE = (224, 224)

def preprocess(file):
    img = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.post("/predict")
async def predict(file: UploadFile):
    image = await file.read()
    x = preprocess(image)
    prob = model.predict(x)[0][0]

    return {
        "label": "dog" if prob > 0.5 else "cat",
        "confidence": float(prob)
    }