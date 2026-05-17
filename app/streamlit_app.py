# ============================================================
# Streamlit App
# Behavioral Cloning Steering Prediction
# ============================================================

import sys
import os

import cv2
import numpy as np
import streamlit as st

from PIL import Image

# ============================================================
# Add project root to path
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

# ============================================================
# Import custom modules
# ============================================================

from src.preprocessing import preprocess_image
from src.model import build_nvidia_model


# ============================================================
# Define paths
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "behavioral_cloning_weights.weights.h5"
)


# ============================================================
# Build model architecture
# ============================================================

model = build_nvidia_model()


# ============================================================
# Load trained weights
# ============================================================

model.load_weights(WEIGHTS_PATH)


# ============================================================
# Streamlit page config
# ============================================================

st.set_page_config(
    page_title="Behavioral Cloning System",
    layout="centered"
)


# ============================================================
# Title
# ============================================================

st.title("Behavioral Cloning Autonomous Driving")

st.write(
    """
    Upload a driving image and predict the steering angle
    using a CNN-based behavioral cloning model.
    """
)


# ============================================================
# File uploader
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a driving image...",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Prediction pipeline
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Load uploaded image
    # --------------------------------------------------------

    image = Image.open(uploaded_file)

    image = np.array(image)

    # --------------------------------------------------------
    # Display original image
    # --------------------------------------------------------

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    processed = preprocess_image(image)

    # --------------------------------------------------------
    # Display processed image
    # --------------------------------------------------------

    st.subheader("Preprocessed Image")

    st.image(
        processed,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    processed_batch = np.expand_dims(
        processed,
        axis=0
    )

    # --------------------------------------------------------
    # Predict steering angle
    # --------------------------------------------------------

    prediction = model.predict(
        processed_batch,
        verbose=0
    )

    steering_angle = prediction[0][0]

    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    st.subheader("Predicted Steering Angle")

    st.success(
        f"{steering_angle:.4f}"
    )

    # --------------------------------------------------------
    # Steering interpretation
    # --------------------------------------------------------

    if steering_angle < -0.05:

        st.info("Suggested Direction: Left Turn")

    elif steering_angle > 0.05:

        st.info("Suggested Direction: Right Turn")

    else:

        st.info("Suggested Direction: Straight")