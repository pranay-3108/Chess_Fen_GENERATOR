# ------------------------------
# Convert 64 labels → FEN string
# ------------------------------

def build_fen_from_labels(labels):
    """
    labels = list of 64 predicted strings: ['empty', 'bK', 'wP', ...]
    Order = row-major (s0-s7 = rank 8, s8-s15 = rank 7, ...)

    Returns: FEN string
    """

    fen_rows = []
    idx = 0

    # Board is predicted from top row → bottom
    for row in range(8):
        empty_count = 0
        fen_row = ""

        for col in range(8):
            label = labels[idx]
            idx += 1

            if label == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0

                # Convert label to FEN symbol
                if label.startswith("w"):
                    fen_row += label[1].upper()
                else:
                    fen_row += label[1].lower()

        if empty_count > 0:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    fen = "/".join(fen_rows)
    return fen
