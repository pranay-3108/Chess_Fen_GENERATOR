import tensorflow as tf
import numpy as np
import cv2
import os
import json

MODEL_PATH = r"C:\chess_board\chess_photoboard_project\models\chess_tile_classifier.h5"
CLASS_MAP_PATH = r"C:\chess_board\chess_photoboard_project\models\class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_MAP_PATH, "r") as f:
    idx_to_class = {int(v): k for k, v in json.load(f).items()}

def predict_tile(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    img = cv2.imread(image_path)
    img = cv2.resize(img, (96, 96))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = np.argmax(preds)
    label = idx_to_class[idx]
    conf = preds[0][idx] * 100

    print("-" * 50)
    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    while True:
        path = input("Enter path of square image: ").strip()
        if path == "":
            break
        predict_tile(path)
