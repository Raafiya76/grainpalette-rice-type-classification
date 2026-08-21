"""
Train the rice-type classifier and save it to models/rice.h5.

This is a cleaned-up, script form of Trainig/rice-classification-1.ipynb.
The notebook's original top layer used ``Dense(10, ...)`` for a 5-class
problem (a leftover bug) - this script fixes it to ``Dense(5, ...)``.

Usage:
    python train_model.py --data-dir Data --epochs 10

The ``Data/`` folder shipped in this repo only has a handful of sample
images per class (enough to sanity-check the pipeline). For a model that
actually generalizes, point --data-dir at the full Kaggle "Rice Image
Dataset" (murat kokludataset/rice-image-dataset), with one sub-folder per
class: Arborio/, Basmati/, Ipsala/, Jasmine/, Karacadag/.
"""
import argparse
import pathlib

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["arborio", "basmati", "ipsala", "jasmine", "karacadag"]
IMG_SIZE = (224, 224)


def load_dataset(data_dir: pathlib.Path):
    # Folder names in Data/ use mixed casing (e.g. "Karakadag" vs "Karacadag"),
    # so match case-insensitively against the canonical class names.
    folders = {p.name.lower(): p for p in data_dir.iterdir() if p.is_dir()}

    X, y = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        folder = folders.get(class_name) or folders.get(class_name.replace("karacadag", "karakadag"))
        if folder is None:
            raise FileNotFoundError(f"No folder found for class '{class_name}' under {data_dir}")
        images = sorted(folder.glob("*"))
        if not images:
            raise FileNotFoundError(f"No images found in {folder}")
        for image_path in images:
            img = cv2.imread(str(image_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, IMG_SIZE)
            X.append(img)
            y.append(label_idx)

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y)
    return X, y


def build_model():
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    top_model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=base_model.output_shape[1:]),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )

    model = tf.keras.Sequential([base_model, top_model])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="Data", help="Path to the labelled image folders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", default="models/rice.h5")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    print(f"Loading images from {data_dir} ...")
    X, y = load_dataset(data_dir)
    print(f"Loaded {len(X)} images across {len(CLASS_NAMES)} classes.")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model()
    model.summary()
    model.fit(X_train, y_train, epochs=args.epochs, validation_data=(X_val, y_val))

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
