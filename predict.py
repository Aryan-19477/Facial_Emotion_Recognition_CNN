
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ==========================
# Load Model
# ==========================

model = load_model("emotion_model.keras")

# ==========================
# Emotion Labels
# ==========================

classes = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

emotion_icons = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}

# ==========================
# Haar Cascade
# ==========================

face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# ==========================
# Prediction History
# ==========================

prediction_history = deque(maxlen=5)

# ==========================
# Face Detection
# ==========================

def detect_faces(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    return faces

# ==========================
# Preprocessing
# ==========================

def preprocess_image(face):

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # CLAHE improves contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    gray = clahe.apply(gray)

    gray = cv2.resize(gray, (48,48))

    gray = gray.astype("float32") / 255.0

    gray = np.expand_dims(gray, axis=-1)

    gray = np.expand_dims(gray, axis=0)

    return gray

# ==========================
# Prediction
# ==========================

def predict_emotion(face):

    processed = preprocess_image(face)

    prediction = model.predict(
        processed,
        verbose=0
    )[0]

    prediction_history.append(prediction)

    avg_prediction = np.mean(
        prediction_history,
        axis=0
    )

    emotion_index = np.argmax(avg_prediction)

    emotion = classes[emotion_index]

    confidence = float(avg_prediction[emotion_index])

    return (
        emotion,
        emotion_icons[emotion],
        confidence,
        avg_prediction
    )
