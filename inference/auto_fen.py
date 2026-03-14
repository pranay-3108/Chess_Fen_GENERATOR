import os
import json
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

# -----------------------
# CONFIG
# -----------------------
MODEL_PATH = r"C:\chess_board\chess_photoboard_project\models\chess_tile_classifier.h5"
IMAGES_DIR = r"C:\chess_board\chess_photoboard_project\images"
OUTPUT_TILES_DIR = r"C:\chess_board\chess_photoboard_project\board_squares"


# -----------------------
# 1. Get latest image
# -----------------------
def get_latest_image():
    files = [
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not files:
        raise FileNotFoundError("❌ No board images found in /images folder.")

    latest = max(files, key=os.path.getmtime)
    print(f"[INFO] Latest board selected: {latest}")
    return latest


# -----------------------
# 2. Auto-rotate if needed
# -----------------------
def auto_rotate(img):
    h, w = img.shape[:2]

    # Rotate if vertical (phone)
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("[INFO] Rotated 90° CCW")

    return img


# -----------------------
# 3. Crop board into 64 tiles
# -----------------------
def crop_board_to_tiles(board_path):
    img = cv2.imread(board_path)
    img = auto_rotate(img)

    H, W = img.shape[:2]

    tile_h = H // 8
    tile_w = W // 8

    os.makedirs(OUTPUT_TILES_DIR, exist_ok=True)

    idx = 0
    for r in range(8):
        for c in range(8):
            y1, y2 = r * tile_h, (r + 1) * tile_h
            x1, x2 = c * tile_w, (c + 1) * tile_w
            tile = img[y1:y2, x1:x2]

            out_path = os.path.join(OUTPUT_TILES_DIR, f"s{idx}.png")
            cv2.imwrite(out_path, tile)
            idx += 1

    if idx != 64:
        raise RuntimeError("❌ Tile count mismatch (expected 64).")

    print("[OK] Saved 64 tiles.")


# -----------------------
# 4. Load CNN
# -----------------------
def load_model():
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[OK] Model loaded.")
    return model


# -----------------------
# 5. Classify each tile
# -----------------------
def classify_tiles(model):
    print("[INFO] Running predictions...")

    # Load class map
    class_indices_path = os.path.join(
        os.path.dirname(MODEL_PATH),
        "class_indices.json"
    )
    class_map = json.load(open(class_indices_path))
    idx_to_label = {v: k for k, v in class_map.items()}

    labels = []

    for i in range(64):
        tile_path = os.path.join(OUTPUT_TILES_DIR, f"s{i}.png")

        if not os.path.exists(tile_path):
            raise FileNotFoundError(f"❌ Missing tile: {tile_path}")

        tile = cv2.imread(tile_path)
        tile = cv2.resize(tile, (128, 128))  # model input size
        tile = tile / 255.0
        tile = np.expand_dims(tile, axis=0)

        pred = model.predict(tile, verbose=0)
        idx = np.argmax(pred)
        labels.append(idx_to_label[idx])

    return labels


# -----------------------
# 6. Build FEN
# -----------------------
from build_fen import build_fen_from_labels


# -----------------------
# MAIN PIPELINE
# -----------------------
def main():
    print("\n========== AUTO FEN START ==========\n")

    board_img = get_latest_image()

    print("[INFO] Cropping board into 64 squares...")
    crop_board_to_tiles(board_img)

    model = load_model()
    labels = classify_tiles(model)

    fen = build_fen_from_labels(labels)

    print("\n====== RESULT FEN ======")
    print(fen)
    print("========================")

    # Save FEN to file
    fen_path = os.path.join(
        os.path.dirname(MODEL_PATH),
        f"fen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(fen_path, "w") as f:
        f.write(fen)

    print(f"[SAVED] FEN saved at: {fen_path}")


if __name__ == "__main__":
    main()
