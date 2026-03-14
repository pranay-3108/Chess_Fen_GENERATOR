# Chess PhotoBoard → FEN Generator

This project converts a chessboard photo into a **FEN (Forsyth–Edwards Notation)** string using a CNN tile classifier.

## Pipeline

Photo → Board Crop → Tile Extraction → CNN Classifier → FEN Generation

## Project Structure

* **inference/** → Run prediction on a board image
* **training/** → Train CNN tile classifier
* **pipeline/** → Full automation pipeline
* **scripts/** → Image preprocessing utilities
* **models/** → Trained model + class indices

## Run Inference

```bash
python inference/run_full_board.py
```

## Model

The project uses a **CNN classifier trained on chessboard tile images**.

Classes predicted:

* wP, wR, wN, wB, wQ, wK
* bP, bR, bN, bB, bQ, bK
* empty

## Output

The system reconstructs the board and produces a **FEN string** describing the full chess position.

Example output:

```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
```

## Future Improvements

* Improve board detection robustness
* Support different board orientations
* Integrate with chess engines for move suggestions

