import sys
import os

import cv2
import numpy as np
import streamlit as st

from PIL import Image

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from src.preprocessing import preprocess_image
from src.model import build_nvidia_model

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "behavioral_cloning_weights.weights.h5"
)

model = build_nvidia_model()

model.load_weights(WEIGHTS_PATH)

st.set_page_config(
    page_title="Behavioral Cloning System",
    layout="centered"
)

st.title("Behavioral Cloning Autonomous Driving")

st.write(
    """
    Upload a driving image and predict the steering angle
    using a CNN-based behavioral cloning model.
    """
)

uploaded_file = st.file_uploader(
    "Choose a driving image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image = np.array(image)

    st.subheader("Original Image")

    st.image(
        image,
        use_container_width=True
    )

    processed = preprocess_image(image)

    st.subheader("Preprocessed Image")

    st.image(
        processed,
        use_container_width=True
    )

    processed_batch = np.expand_dims(
        processed,
        axis=0
    )

    prediction = model.predict(
        processed_batch,
        verbose=0
    )

    steering_angle = prediction[0][0]

    st.subheader("Predicted Steering Angle")

    st.success(
        f"{steering_angle:.4f}"
    )

    if steering_angle < -0.05:

        st.info("Suggested Direction: Left Turn")

    elif steering_angle > 0.05:

        st.info("Suggested Direction: Right Turn")

    else:

        st.info("Suggested Direction: Straight")
