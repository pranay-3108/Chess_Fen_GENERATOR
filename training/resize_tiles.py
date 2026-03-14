import os
import cv2

# ===== CONFIG =====
PROJECT_ROOT = r"C:\chess_board\chess_photoboard_project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

TARGET_SIZE = (128, 128)
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def resize_folder(folder):
    if not os.path.isdir(folder):
        print(f"[WARN] Folder not found: {folder}")
        return

    for cls in sorted(os.listdir(folder)):
        cls_path = os.path.join(folder, cls)
        if not os.path.isdir(cls_path):
            continue

        print(f"[CLASS] {cls}")
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            _, ext = os.path.splitext(fname)
            if ext.lower() not in VALID_EXT:
                continue

            img = cv2.imread(fpath)
            if img is None:
                print(f"  [SKIP] unreadable: {fpath}")
                continue

            img_resized = cv2.resize(img, TARGET_SIZE)
            cv2.imwrite(fpath, img_resized)

        print(f"  [OK] resized images in {cls_path}")


def main():
    print("[INFO] Project root:", PROJECT_ROOT)
    print("[STEP] Resizing TRAIN tiles...")
    resize_folder(TRAIN_DIR)
    print("[STEP] Resizing VAL tiles...")
    resize_folder(VAL_DIR)
    print("[DONE] All tiles normalized to", TARGET_SIZE)


if __name__ == "__main__":
    main()

