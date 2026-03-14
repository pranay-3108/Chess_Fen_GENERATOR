import os
import json
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ===== PATHS =====
PROJECT_ROOT = r"C:\chess_board\chess_photoboard_project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "chess_tile_classifier.h5")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")

# ===== HYPERPARAMS =====
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20  # early stopping will stop earlier

# ===== DATA GENERATORS =====

train_aug = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    brightness_range=[0.7, 1.3],
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_aug = ImageDataGenerator(
    rescale=1.0 / 255.0
)

train_gen = train_aug.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_aug.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_gen.num_classes
print("[INFO] Classes:", train_gen.class_indices)

# Save class indices mapping
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print("[INFO] Saved class indices to:", CLASS_INDICES_PATH)


# ===== CLASS WEIGHTS (handle empty >> others) =====

def compute_class_weights(generator):
    counts = Counter(generator.classes)
    max_count = max(counts.values())
    class_weights = {cls_idx: max_count / float(count) for cls_idx, count in counts.items()}
    return class_weights


class_weights = compute_class_weights(train_gen)
print("[INFO] Class weights:", class_weights)


# ===== BUILD MODEL (MobileNetV2 backbone) =====

def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=13):
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base.trainable = False  # freeze backbone

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model


model = build_model(num_classes=num_classes)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ===== CALLBACKS =====

checkpoint_cb = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ===== TRAIN =====

steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
validation_steps = max(1, val_gen.samples // BATCH_SIZE)

print("[INFO] Starting training...")
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps,
    class_weight=class_weights,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

print("[DONE] Training complete.")
print("[BEST] Model saved at:", MODEL_PATH)

