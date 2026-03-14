import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Optional: advanced metrics
try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] scikit-learn not installed. Only accuracy will be shown.")

PROJECT_ROOT = r"C:\chess_board\chess_photoboard_project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "chess_tile_classifier.h5")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

print("[INFO] Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
print("[OK] Model loaded.")

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
print("[INFO] Classes:", class_names)

datagen = ImageDataGenerator(rescale=1.0 / 255.0)

val_gen = datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("[INFO] Evaluating...")
loss, acc = model.evaluate(val_gen, verbose=1)
print("\n==============================")
print("  VALIDATION METRICS")
print("==============================")
print(f"Val Loss     : {loss:.4f}")
print(f"Val Accuracy : {acc * 100:.2f}%")

if SKLEARN_AVAILABLE:
    print("\n[INFO] Predicting for detailed report...")
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))
else:
    print("\n[NOTE] Install scikit-learn for per-class metrics:")
    print("  pip install scikit-learn")
