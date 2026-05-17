# ============================================================
# augmentation.py
# Data augmentation functions for behavioral cloning
# ============================================================

import cv2
import numpy as np
import random


# ============================================================
# 1. RANDOM FLIP
# ============================================================

def random_flip(image, steering_angle):

    """
    Randomly flip image horizontally.

    Steering angle must also be inverted.
    """

    if random.random() < 0.5:

        image = cv2.flip(image, 1)

        steering_angle = -steering_angle

    return image, steering_angle


# ============================================================
# 2. RANDOM BRIGHTNESS
# ============================================================

def random_brightness(image):

    """
    Randomly adjust image brightness.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    brightness_scale = 1.0 + (np.random.rand() - 0.5) * 0.4

    hsv[:,:,2] = hsv[:,:,2] * brightness_scale

    hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)

    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image


# ============================================================
# 3. RANDOM SHADOW
# ============================================================

def random_shadow(image):

    """
    Add random shadow to simulate lighting variation.
    """

    height, width, _ = image.shape

    x1, y1 = width * np.random.rand(), 0
    x2, y2 = width * np.random.rand(), height

    xm, ym = np.mgrid[0:height, 0:width]

    mask = np.zeros_like(image[:,:,1])

    mask[
        (ym - y1) * (x2 - x1)
        - (y2 - y1) * (xm - x1) > 0
    ] = 1

    shadow_intensity = 0.5

    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    image_hls[:,:,1][mask == 1] *= shadow_intensity

    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return image


# ============================================================
# 4. RANDOM TRANSLATION (PAN)
# ============================================================

def random_translate(image, steering_angle, range_x=100, range_y=10):

    """
    Randomly shift image horizontally and vertically.

    Horizontal movement affects steering angle.
    """

    trans_x = range_x * (np.random.rand() - 0.5)

    trans_y = range_y * (np.random.rand() - 0.5)

    steering_angle += trans_x * 0.002

    translation_matrix = np.float32([
        [1, 0, trans_x],
        [0, 1, trans_y]
    ])

    height, width = image.shape[:2]

    image = cv2.warpAffine(
        image,
        translation_matrix,
        (width, height)
    )

    return image, steering_angle


# ============================================================
# 5. RANDOM ZOOM
# ============================================================

def random_zoom(image):

    """
    Randomly zoom into image.
    """

    zoom_scale = 1 + (np.random.rand() * 0.3)

    height, width = image.shape[:2]

    new_height = int(height / zoom_scale)

    new_width = int(width / zoom_scale)

    y1 = np.random.randint(0, height - new_height)

    x1 = np.random.randint(0, width - new_width)

    cropped = image[
        y1:y1+new_height,
        x1:x1+new_width
    ]

    image = cv2.resize(cropped, (width, height))

    return image


# ============================================================
# 6. FULL AUGMENTATION PIPELINE
# ============================================================

def augment_image(image, steering_angle):

    """
    Apply all augmentation techniques.
    """

    image, steering_angle = random_flip(
        image,
        steering_angle
    )

    image, steering_angle = random_translate(
        image,
        steering_angle
    )

    image = random_brightness(image)

    image = random_shadow(image)

    image = random_zoom(image)

    return image, steering_angle