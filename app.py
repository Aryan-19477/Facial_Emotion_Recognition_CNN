import streamlit as st
import cv2
import numpy as np
from PIL import Image
import av
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

from predict import (
    detect_faces,
    predict_emotion,
    classes
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="centered"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("😊 Facial Emotion Recognition")

st.caption(
    "Real-time facial emotion recognition using a Convolutional Neural Network (CNN) trained on the FER2013 dataset."
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📌 About Model")

    st.write("**Model:** CNN with Batch Normalisation and Dropout")

    st.write("**Dataset:** FER2013")

    st.write("**Classes:** 7")

    st.write("**Image Size:** 48 × 48")

    st.write("**Framework:** TensorFlow / Keras")

    st.write("**Face Detection:** OpenCV Haar Cascade")

    st.divider()

    st.write("### Supported Emotions")

    for emotion in classes:
        st.write(f"• {emotion.capitalize()}")

# --------------------------------------------------
# INPUT MODE
# --------------------------------------------------

mode = st.radio(
    "Choose Input Method",
    ["📸 Capture Photo", "🎥 Live Webcam"],
    horizontal=True
)

st.divider()

# ==================================================
# CAPTURE PHOTO
# ==================================================

if mode == "📸 Capture Photo":

    picture = st.camera_input("📸 Capture Image")

    if picture:

        with st.spinner("Predicting emotion..."):

            image = Image.open(picture)

            image = np.array(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            faces = detect_faces(image)

            if len(faces) == 0:

                st.warning("No face detected. Please capture the image again.")

            else:

                x, y, w, h = faces[0]

                face = image[y:y+h, x:x+w]

                emotion, emoji, confidence, probs = predict_emotion(face)

                cv2.rectangle(
                    image,
                    (x, y),
                    (x+w, y+h),
                    (0,255,0),
                    2
                )

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                st.image(
                    image,
                    use_container_width=True
                )

                st.subheader("Prediction")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Emotion",
                        f"{emoji} {emotion}"
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        f"{confidence*100:.2f}%"
                    )

                st.divider()

                st.subheader("Class Probabilities")

                for label, p in zip(classes, probs):

                    st.write(f"**{label.capitalize()}**")

                    st.progress(float(p))

                    st.caption(f"{p*100:.2f}%")

# ==================================================
# LIVE WEBCAM
# ==================================================

else:
    RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": ["stun:stun.l.google.com:19302"]
            }
        ]
    }
)

    class EmotionProcessor(VideoProcessorBase):

        def __init__(self):

            self.frame_count = 0

            self.last_emotion = ""
            
            

            self.last_emoji = ""

            self.last_confidence = 0

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")

            self.frame_count += 1

            faces = detect_faces(img)

            for (x, y, w, h) in faces:

                face = img[y:y+h, x:x+w]

                if self.frame_count % 3 == 0:

                    emotion, emoji, confidence, probs = predict_emotion(face)

                    self.last_emotion = emotion

                    self.last_emoji = emoji

                    self.last_confidence = confidence

                cv2.rectangle(
                    img,
                    (x, y),
                    (x+w, y+h),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    img,
                    f"{self.last_emoji} {self.last_emotion} ({self.last_confidence*100:.1f}%)",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

            return av.VideoFrame.from_ndarray(
                img,
                format="bgr24"
            )

    webrtc_streamer(

        key="emotion",

        video_processor_factory=EmotionProcessor,
         rtc_configuration=RTC_CONFIGURATION,

        media_stream_constraints={

            "video":{

                "width":640,

                "height":480,

                "frameRate":20

            },

            "audio":False

        },

        async_processing=True

    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built using TensorFlow • OpenCV •  Haar Cascade • Streamlit"
)
