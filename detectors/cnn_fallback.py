"""
Optional kbulutozler CNN ear-landmark fallback.

Repo: https://github.com/kbulutozler/ear-landmark-detection-with-CNN
Input:  224×224×3 (Keras ImageNet preprocess_input)
Output: 110 values → 55 (x, y)  [all x then all y]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


CNN_HELP = """
========================================================================
CNN fallback weights missing or invalid.

Required file:
  {path}

Obtain via Git LFS from:
  https://github.com/kbulutozler/ear-landmark-detection-with-CNN
  file: my_model.h5  (~108 MB)

Install TensorFlow first:  pip install tensorflow
========================================================================
"""


def _build_architecture():
    """Keras Sequential matching my_CNN_model.py."""
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        MaxPooling2D,
    )
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(
        Conv2D(
            16,
            (3, 3),
            input_shape=(224, 224, 3),
            kernel_initializer="random_uniform",
            activation="relu",
        )
    )
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (5, 5), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(110))
    return model


class CNNEarLandmarker:
    """kbulutozler Keras CNN: 224×224 → 55 landmarks."""

    DISPLAY_NAME = "kbulutozler-CNN"

    def __init__(self, weights_path: str, input_size: int = 224) -> None:
        self.input_size = input_size
        path = Path(weights_path)
        if not path.is_file() or path.stat().st_size < 10_000:
            # Git LFS pointer files are tiny (~130 bytes)
            raise FileNotFoundError(CNN_HELP.format(path=path.resolve()))

        try:
            from tensorflow.keras.models import load_model
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for --model cnn. Install with: pip install tensorflow"
            ) from exc

        try:
            self.model = load_model(str(path), compile=False)
        except Exception:
            # Architecture + load_weights fallback
            self.model = _build_architecture()
            self.model.load_weights(str(path))

        print(f"[CNN] Loaded {path}")

    def predict(self, ear_bgr: np.ndarray) -> np.ndarray:
        from tensorflow.keras.applications.imagenet_utils import preprocess_input

        img = cv2.resize(ear_bgr, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = np.expand_dims(rgb, axis=0)
        x = preprocess_input(x)
        pred = self.model.predict(x, verbose=0)[0]
        # Output layout in utilities.py: all x then all y
        xs = pred[:55]
        ys = pred[55:110]
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        return pts
