import cv2
import numpy as np
import os

OUTPUT_DIR = r"C:\chess_board\chess_photoboard_project\board_squares"

# ---- Mouse Callback to Select Corners ----
points = []

def select_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 2 Corners", param)


def crop_board_to_tiles_manual(img_path):
    global points
    points = []

    img = cv2.imread(img_path)
    clone = img.copy()

    cv2.imshow("Select 2 Corners", clone)
    cv2.setMouseCallback("Select 2 Corners", select_point, clone)

    print("\n[INFO] Click TOP-LEFT corner of board")
    print("[INFO] Then click BOTTOM-RIGHT corner of board\n")

    # Wait until 2 clicks
    while True:
        cv2.imshow("Select 2 Corners", clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(points) == 2:
            break

    cv2.destroyAllWindows()

    if len(points) != 2:
        raise RuntimeError("❌ You must select exactly 2 points.")

    (x1, y1), (x2, y2) = points

    # Ensure correct ordering
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    board = img[y_min:y_max, x_min:x_max]

    # Tile sizes
    H, W = board.shape[:2]
    tile_h = H // 8
    tile_w = W // 8

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    idx = 0
    for r in range(8):
        for c in range(8):
            y1_t = r * tile_h
            x1_t = c * tile_w
            tile = board[y1_t:y1_t + tile_h, x1_t:x1_t + tile_w]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"s{idx}.png"), tile)
            idx += 1

    print(f"[OK] Saved {idx} tiles to {OUTPUT_DIR}")
