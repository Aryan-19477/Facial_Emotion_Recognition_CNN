# 😊 Facial Emotion Recognition using CNN

A real-time **Facial Emotion Recognition** web application built using **TensorFlow, Keras, OpenCV, and Streamlit**. The model is trained from scratch on the **FER2013** dataset using a Convolutional Neural Network (CNN) and deployed on **Render** for online access.

---

# 📌 Project Overview

This application detects a human face and predicts one of the following seven emotions:

- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😄 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

The web application provides three different input methods:

- 📸 Capture Photo
- 🎥 Live Webcam
- 🖼 Upload Image

The predicted emotion is displayed along with its confidence score.

---

# 🚀 Features

- Real-time facial emotion recognition
- Live webcam prediction
- Capture image directly from camera
- Upload image from device
- OpenCV Haar Cascade face detection
- CNN trained from scratch
- Probability scores for all seven emotions
- Clean and responsive Streamlit interface
- Deployed on Render

---

# 🛠 Tech Stack

### Language

- Python

### Deep Learning

- TensorFlow
- Keras

### Computer Vision

- OpenCV

### Web Framework

- Streamlit

### Dataset

- FER2013

### Deployment

- Render

---

# 📂 Project Structure

```
Facial_Emotion_Recognition_CNN/

│
├── app.py
├── predict.py
├── emotion_model.keras
├── render.yaml
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── README.md
└── Facial_Emotion_Recognition_CNN.ipynb
```

---

# 🧠 Model Development

The objective of this project was not only to build a facial emotion recognition model but also to understand the complete deep learning workflow by improving the model step by step.

## Phase 1 – Baseline CNN

The first model was built using a simple CNN architecture consisting of:

- Convolution Layers
- Max Pooling Layers
- Flatten Layer
- Dense Output Layer

This baseline model served as the starting point for further improvements.

---

## Phase 2 – Model Improvements

The CNN architecture was gradually improved using multiple deep learning techniques such as:

- Additional Convolution Layers
- Batch Normalization
- Dropout
- Data Normalization
- Improved Optimizer Configuration
- Learning Rate Scheduling
- Early Stopping
- Model Checkpointing
- ReduceLROnPlateau Callback

These enhancements improved model stability and reduced overfitting.

---

# 📊 Dataset

The project uses the **FER2013** dataset.

Dataset Details:

- 35,887 facial images
- 48 × 48 grayscale images
- 7 emotion classes

The dataset contains facial expressions captured under different lighting conditions, poses, and facial variations.

---

# 🏗 CNN Architecture

The final model consists of:

- Input Layer
- Multiple Convolution Layers
- Batch Normalization
- Max Pooling Layers
- Dropout Layers
- Dense Layers
- Softmax Output Layer

---

# 📈 Training Workflow

1. Load FER2013 dataset
2. Image preprocessing
3. Data augmentation
4. Build CNN model
5. Train the model
6. Evaluate performance
7. Save trained model
8. Build Streamlit application
9. Deploy on Render

---

# 💻 Application Workflow

```
User Input
   │
   ├──────────────┐──────────────┐
   │              │              │
Capture Photo   Live Webcam   Upload Image
        │            │              │
        └────────────┴──────────────┘
                     │
                     ▼
          Face Detection (OpenCV)
                     │
                     ▼
          Image Preprocessing
                     │
                     ▼
             CNN Prediction
                     │
                     ▼
     Emotion + Confidence Score
                     │
                     ▼
          Streamlit User Interface
```

---

# 🌐 Deployment

The application was first developed and tested locally using Streamlit.

Run locally:

```bash
streamlit run app.py
```

After testing, the project was uploaded to GitHub and deployed using **Render**, making the application accessible through any web browser without requiring local installation.

Deployment Pipeline:

```
Local Development
        │
        ▼
GitHub Repository
        │
        ▼
Render
        │
        ▼
Live Web Application
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/Aryan-19477/Facial_Emotion_Recognition_CNN.git
```

Move to the project directory

```bash
cd Facial_Emotion_Recognition_CNN
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📸 Application Features

The application supports:

- 📸 Capture Photo
- 🎥 Live Webcam
- 🖼 Upload Image

For every prediction, the application displays:

- Detected Face
- Predicted Emotion
- Confidence Score
- Probability Distribution of all seven emotions

---

# 📌 Future Improvements

Possible future enhancements include:

- Higher accuracy using Transfer Learning
- Multiple face recognition
- Faster real-time inference
- Mobile application deployment
- Emotion tracking over time
- Improved face detection models

---

# 📚 Learning Outcomes

This project helped me gain practical experience in:

- Convolutional Neural Networks (CNN)
- TensorFlow & Keras
- Image preprocessing
- OpenCV
- Streamlit
- Real-time inference
- Deep learning model optimization
- Model deployment using Render
- End-to-end machine learning project development

---

# 👨‍💻 Author

**Aryan Agrawal**

B.Tech in Computer Science and Engineering  
IIIT Nagpur

---

# ⭐ If you found this project useful, consider giving it a Star.
