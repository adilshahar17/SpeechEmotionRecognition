# SpeechEmotionRecognition
A system to detect emotions (e.g., happiness, sadness, anger) from speech recordings by analyzing vocal features.
This project implements Speech Emotion Recognition (SER) using the RAVDESS dataset. It explores both traditional machine learning (RandomForest) and deep learning (CNN) approaches to classify emotions from audio recordings.

# Dataset
- Source: RAVDESS Emotional Speech Audio Dataset (kaggle.com)
- Contains recordings labeled with 8 emotions:
- Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

# Workflow
- Data Download
    Dataset is fetched using kagglehub.
- Feature Extraction
    - Audio files processed with librosa.
    - Extracted MFCC features (40 coefficients) averaged across time.
    - Labels mapped from file naming convention.
- Baseline Model (RandomForest)
    - Trained on MFCC features.
    - Evaluated with classification report and confusion matrix.
- Deep Learning Model (CNN)
    - Input: MFCC features reshaped for convolutional layers.
    - Architecture:
      - Conv1D → BatchNorm → MaxPooling
      - Conv1D → BatchNorm → MaxPooling
      - Dense → Dropout → Softmax output
    - Trained with Adam optimizer, sparse_categorical_crossentropy loss.
    - Achieved improved accuracy compared to baseline.
- Evaluation
    - Accuracy and loss curves plotted.
    - Confusion matrix visualized for baseline.
    - CNN model saved as emotion_model.h5 with class labels in classes.npy.

# Usage in Colab
Predict Emotion from Audio

    import librosa
    import numpy as np
    import tensorflow as tf
    model = tf.keras.models.load_model("emotion_model.h5")
    classes = np.load("classes.npy", allow_pickle=True)
    def predict_emotion(file_path):
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        mfccs = mfccs.reshape(1, mfccs.shape[0], 1)
        preds = model.predict(mfccs)
        pred_class = classes[np.argmax(preds)]
        confidence = np.max(preds)
        return pred_class, confidence
    
    test_file = "/content/sample_data/03-01-04-02-02-02-03.wav"   # enter the sample file
    emotion, conf = predict_emotion(test_file)
    print(f"Predicted Emotion: {emotion} (Confidence: {conf:.2f})")

# Streamlit Demo
You can run a simple web app to test emotion recognition interactively.
serapp.py

    import streamlit as st
    import librosa
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import LabelEncoder
    import os
    
    model = load_model("emotion_model.h5") 
    encoder = LabelEncoder()
    encoder.classes_ = np.load("classes.npy", allow_pickle=True)  
    
    def extract_features(file_path):
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs
    
    st.title("🎤 Speech Emotion Recognition Demo")
    st.write("Upload a speech audio file (.wav) and get the predicted emotion.")
    
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])
    
    if uploaded_file is not None:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
    
        features = extract_features("temp.wav").reshape(1, -1, 1)
    
        prediction = model.predict(features)
        emotion = encoder.inverse_transform([np.argmax(prediction)])
    
        st.success(f"Predicted Emotion: **{emotion[0]}**")
    
        y, sr = librosa.load("temp.wav", duration=3, offset=0.5)
        st.write("Spectrogram:")
        st.audio(uploaded_file, format="audio/wav")

Run the app
streamlit run app.py

# Requirements
  - Python 3.8+
  - Libraries: librosa, numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow, kagglehub, streamlit

# Results
  - Baseline RandomForest: ~60–70% accuracy
  - CNN Deep Learning: Higher accuracy (~75–85% depending on dataset size and balance)

# Future Improvements
  - Larger datasets for better generalization
  - Advanced architectures (CNN+LSTM, Transformers)
  - Real-time deployment with microphone streaming
  - Multilingual emotion recognition




    
