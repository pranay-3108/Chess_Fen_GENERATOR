import os
import cv2
import json
import numpy as np
import tensorflow as tf


def classify_all_tiles(model_path, tile_dir):
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(model_path)

    class_map_path = os.path.join(os.path.dirname(model_path), "class_indices.json")
    with open(class_map_path, "r") as f:
        class_map = json.load(f)

    idx_to_label = {v: k for k, v in class_map.items()}

    labels = []

    for i in range(64):
        tile_path = os.path.join(tile_dir, f"s{i}.png")
        if not os.path.exists(tile_path):
            raise FileNotFoundError(f"Missing tile: {tile_path}")

        tile = cv2.imread(tile_path)
        tile = cv2.resize(tile, (128, 128)) / 255.0
        tile = np.expand_dims(tile, axis=0)

        pred = model.predict(tile, verbose=0)
        idx = np.argmax(pred)
        labels.append(idx_to_label[idx])

    print("[OK] Tile classification complete!")
    return labels


# Manual CLI mode
if __name__ == "__main__":
    import sys
    from crop_board_to_tiles import crop_board_to_tiles
    from build_fen import build_fen_from_labels

    if len(sys.argv) < 2:
        print("Usage: python run_full_board.py your_image.jpg")
        exit()

    img_path = sys.argv[1]
    model_path = r"C:\chess_board\chess_photoboard_project\models\chess_tile_classifier.h5"
    tile_dir = r"C:\chess_board\chess_photoboard_project\board_squares"

    crop_board_to_tiles(img_path)
    labels = classify_all_tiles(model_path, tile_dir)
    fen = build_fen_from_labels(labels)

    print("\nFINAL FEN:", fen)
