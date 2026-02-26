import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# Load trained model and label encoder
model = load_model("emotion_model.h5")   # Save your trained model as emotion_model.h5
encoder = LabelEncoder()
encoder.classes_ = np.load("classes.npy", allow_pickle=True)  # Save encoder classes after training

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

# Streamlit UI
st.title("🎤 Speech Emotion Recognition Demo")
st.write("Upload a speech audio file (.wav) and get the predicted emotion.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features
    features = extract_features("temp.wav").reshape(1, -1, 1)

    # Predict emotion
    prediction = model.predict(features)
    emotion = encoder.inverse_transform([np.argmax(prediction)])

    # Display result
    st.success(f"Predicted Emotion: **{emotion[0]}**")

    # Optional: show spectrogram
    y, sr = librosa.load("temp.wav", duration=3, offset=0.5)
    st.write("Spectrogram:")
    st.audio(uploaded_file, format="audio/wav")
    st.pyplot(librosa.display.specshow(librosa.feature.melspectrogram(y=y, sr=sr), sr=sr))