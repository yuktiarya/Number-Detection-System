from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cnn_mnist.keras"
MAX_IMAGE_BYTES = 5 * 1024 * 1024

app = FastAPI(title="Digit Studio", version="1.0.0")

model = None


def load_model():
    global model
    if model is None and MODEL_PATH.exists():
        model = keras.models.load_model(MODEL_PATH, compile=False)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.grayscale(image)
    image.thumbnail((4096, 4096))
    pixels = np.asarray(image, dtype=np.uint8)

    if pixels.mean() > 127:
        pixels = 255 - pixels

    foreground = pixels > 24
    if foreground.any():
        rows, columns = np.where(foreground)
        cropped = Image.fromarray(pixels[rows.min():rows.max() + 1, columns.min():columns.max() + 1])
        cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)

        canvas = Image.new("L", (28, 28), 0)
        x_offset = (28 - cropped.width) // 2
        y_offset = (28 - cropped.height) // 2
        canvas.paste(cropped, (x_offset, y_offset))
        pixels = np.asarray(canvas, dtype=np.uint8)
    else:
        pixels = np.zeros((28, 28), dtype=np.uint8)

    return (pixels.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)


@app.get("/health")
def health():
    loaded_model = load_model()
    return {"status": "ok", "model_loaded": loaded_model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/webp", "image/bmp"}:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, WEBP, or BMP image.")

    contents = await file.read(MAX_IMAGE_BYTES + 1)
    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller.")

    try:
        with Image.open(BytesIO(contents)) as image:
            if image.width > 4096 or image.height > 4096 or image.width * image.height > 16_000_000:
                raise HTTPException(status_code=413, detail="Image dimensions must be at most 4096 pixels per side.")
            image.load()
            input_image = image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc

    loaded_model = load_model()
    if loaded_model is None:
        raise HTTPException(
            status_code=503,
            detail="The model is not available. Run `python train_model.py` first.",
        )

    scores = np.asarray(loaded_model.predict(preprocess_image(input_image), verbose=0))[0]
    if scores.shape != (10,):
        raise HTTPException(status_code=500, detail="The model must return scores for digits 0 through 9.")

    digit = int(np.argmax(scores))
    return {
        "digit": digit,
        "confidence": float(scores[digit]),
        "probabilities": [float(score) for score in scores],
    }
