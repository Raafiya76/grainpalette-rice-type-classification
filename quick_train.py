#!/usr/bin/env python
"""Quick training script to generate the model."""
import os
import sys
import pathlib
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF verbosity

CLASS_NAMES = ["arborio", "basmati", "ipsala", "jasmine", "karacadag"]
IMG_SIZE = (224, 224)
DATA_DIR = pathlib.Path("Data")
OUTPUT_PATH = pathlib.Path("models/rice.h5")

print("[1/4] Loading dataset...")
X, y = [], []
folders = {p.name.lower(): p for p in DATA_DIR.iterdir() if p.is_dir()}

for label_idx, class_name in enumerate(CLASS_NAMES):
    folder = folders.get(class_name) or folders.get(class_name.replace("karacadag", "karakadag"))
    if folder is None:
        print(f"ERROR: No folder for {class_name}")
        sys.exit(1)
    
    for image_path in sorted(folder.glob("*")):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        X.append(img)
        y.append(label_idx)

X = np.array(X, dtype="float32") / 255.0
y = np.array(y)
print(f"✓ Loaded {len(X)} images")

print("[2/4] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}")

print("[3/4] Building and training model...")
base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
top_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=base_model.output_shape[1:]),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
])
model = tf.keras.Sequential([base_model, top_model])
model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

# Train with minimal output
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), verbose=1)

print("[4/4] Saving model...")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(OUTPUT_PATH)
print(f"✓ Model saved to {OUTPUT_PATH}")
print("✓ Training complete! Refresh the web app to use the model.")
