# ============================================================
# preprocessing.py
# Image preprocessing functions for behavioral cloning
# ============================================================

import cv2
import numpy as np


def preprocess_image(image):

    """
    Preprocess driving image for CNN model.

    Steps:
    1. Crop unnecessary regions
    2. Resize image
    3. Convert RGB to YUV
    4. Apply Gaussian blur
    5. Normalize pixel values
    """

    # ------------------------------------------------
    # Crop image
    # Removes sky and vehicle hood
    # ------------------------------------------------

    cropped = image[60:135, :, :]

    # ------------------------------------------------
    # Resize image
    # NVIDIA model expects 200x66
    # ------------------------------------------------

    resized = cv2.resize(cropped, (200, 66))

    # ------------------------------------------------
    # Convert RGB to YUV
    # Better for feature extraction
    # ------------------------------------------------

    yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)

    # ------------------------------------------------
    # Apply Gaussian Blur
    # Reduces image noise
    # ------------------------------------------------

    blurred = cv2.GaussianBlur(yuv, (3,3), 0)

    # ------------------------------------------------
    # Normalize image
    # Pixel range: 0-255 → 0-1
    # ------------------------------------------------

    normalized = blurred / 255.0

    return normalized