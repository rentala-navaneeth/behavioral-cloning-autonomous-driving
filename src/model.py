# ============================================================
# model.py
# NVIDIA CNN Behavioral Cloning Model
# ============================================================

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Dropout
)


def build_nvidia_model():

    """
    Build NVIDIA behavioral cloning CNN model.
    """

    model = Sequential()

    # =====================================================
    # Convolution Layers
    # =====================================================

    model.add(
        Conv2D(
            24,
            (5,5),
            strides=(2,2),
            activation='relu',
            input_shape=(66, 200, 3)
        )
    )

    model.add(
        Conv2D(
            36,
            (5,5),
            strides=(2,2),
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            48,
            (5,5),
            strides=(2,2),
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            64,
            (3,3),
            activation='relu'
        )
    )

    model.add(
        Conv2D(
            64,
            (3,3),
            activation='relu'
        )
    )

    # =====================================================
    # Flatten Layer
    # =====================================================

    model.add(Flatten())

    # =====================================================
    # Fully Connected Layers
    # =====================================================

    model.add(Dense(100, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(10, activation='relu'))

    # =====================================================
    # Regression Output Layer
    # =====================================================

    model.add(Dense(1))

    return model