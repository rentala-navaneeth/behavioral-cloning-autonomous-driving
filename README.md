# Behavioral Cloning System for Autonomous Driving

An end-to-end deep learning project that predicts steering angles directly from front-facing driving images using a Convolutional Neural Network (CNN)-based behavioral cloning approach.

This project was built using TensorFlow/Keras, OpenCV, and Streamlit to simulate autonomous steering prediction behavior from visual road input.

---

# Project Overview

Behavioral cloning is a supervised learning approach where a model learns driving behavior from human driving data.

The system learns a mapping:

Driving Image → Steering Angle

using a CNN regression model trained on driving simulator data from the Udacity Self-Driving Car dataset.

The project includes:

- Image preprocessing pipeline
- Data augmentation pipeline
- NVIDIA-inspired CNN architecture
- Steering-angle regression training
- Model evaluation and visualization
- Real-time steering prediction
- Streamlit deployment interface

---

# Dataset

Dataset Used:
- Udacity Self-Driving Car Behavioral Cloning Dataset

Dataset Includes:
- Center camera images
- Left and right camera images
- Steering angle labels
- Driving logs

Target Variable:
- Steering Angle (continuous regression output)

Dataset Size:
- ~3400 driving samples

---

# Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

---

# Project Structure

```bash
behavioral cloning autonomous driving/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── self_driving_car_dataset_jungle/
│
├── models/
│   ├── behavioral_cloning_model.keras
│   └── behavioral_cloning_weights.weights.h5
│
├── notebooks/
│   └── behavioral_cloning_eda.ipynb
│
├── src/
│   ├── augmentation.py
│   ├── evaluate.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Data Preprocessing Pipeline

The preprocessing pipeline was implemented using OpenCV to improve CNN learning efficiency and reduce irrelevant visual noise.

Preprocessing Steps:
- Cropping unnecessary regions (sky and vehicle hood)
- Image resizing to NVIDIA architecture dimensions
- RGB → YUV color space conversion
- Gaussian blur
- Pixel normalization

Final Input Shape:
- (66, 200, 3)

---

# Data Augmentation Pipeline

To reduce overfitting and improve generalization, multiple augmentation techniques were implemented:

- Horizontal flipping
- Steering angle correction
- Brightness adjustment
- Random shadow generation
- Random zoom
- Random translation (pan)

These augmentations simulate:
- Different lighting conditions
- Recovery driving behavior
- Road position variation

---

# Model Architecture

The project uses an NVIDIA-inspired CNN behavioral cloning architecture.

Architecture Includes:
- Convolutional layers for feature extraction
- ReLU activations
- Fully connected dense layers
- Dropout regularization
- Regression output layer

Output:
- Continuous steering angle prediction

Total Parameters:
- ~252K trainable parameters

Loss Function:
- Mean Squared Error (MSE)

Optimizer:
- Adam Optimizer

---

# Training Pipeline

The model training workflow includes:

- Train-validation split
- Batch generators
- Dynamic augmentation during training
- Preprocessing integration
- EarlyStopping callback
- ModelCheckpoint saving

Training was performed using TensorFlow/Keras on Google Colab with GPU acceleration.

---

# Results

Best Validation Loss:
- ~0.15 validation MSE loss

Observations:
- Model successfully learned steering-angle prediction behavior
- Data augmentation improved generalization capability
- EarlyStopping prevented overfitting
- The model demonstrated reasonable steering predictions on unseen driving images

---

# Streamlit Deployment

A lightweight Streamlit application was developed for real-time steering prediction.

Features:
- Upload driving images
- Visualize preprocessing output
- Predict steering angle
- Display steering direction interpretation

Run Streamlit App:

```bash
streamlit run app/streamlit_app.py
```

---

# Sample Workflow

1. Upload driving image
2. Apply preprocessing pipeline
3. Feed processed image into CNN
4. Predict steering angle
5. Display steering direction

---

# Key Learning Outcomes

This project demonstrates:
- Deep learning for computer vision
- CNN-based regression modeling
- Behavioral cloning systems
- Image preprocessing techniques
- Data augmentation strategies
- TensorFlow/Keras model training
- Real-time inference deployment
- End-to-end ML workflow development

---

# Future Improvements

Potential future enhancements include:
- Temporal sequence modeling using LSTMs
- Video-based inference
- Larger driving datasets
- Simulator integration
- Advanced steering smoothing techniques

---
