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

    cropped = image[60:135, :, :]

    resized = cv2.resize(cropped, (200, 66))

    yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)

    blurred = cv2.GaussianBlur(yuv, (3,3), 0)

    normalized = blurred / 255.0

    return normalized
