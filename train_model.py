from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "mnist_train.csv"
TEST_CSV = BASE_DIR / "mnist_test.csv"
MODEL_PATH = BASE_DIR / "cnn_mnist.keras"


def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    if "label" not in frame.columns:
        raise ValueError(f"{path.name} must contain a 'label' column.")

    features = frame.drop(columns=["label"]).to_numpy(dtype=np.float32)
    labels = frame["label"].to_numpy(dtype=np.int64)
    if features.shape[1] != 28 * 28:
        raise ValueError(f"{path.name} must contain exactly 784 pixel columns.")
    if not np.isin(labels, np.arange(10)).all():
        raise ValueError(f"{path.name} labels must be digits from 0 to 9.")

    images = (features / 255.0).reshape(-1, 28, 28, 1)
    return images, keras.utils.to_categorical(labels, num_classes=10)


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    train_images, train_labels = load_dataset(TRAIN_CSV)
    test_images, test_labels = load_dataset(TEST_CSV)

    model = build_model()
    model.fit(
        train_images,
        train_labels,
        epochs=5,
        batch_size=32,
        validation_data=(test_images, test_labels),
        verbose=2,
    )
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
    model.save(MODEL_PATH)
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
