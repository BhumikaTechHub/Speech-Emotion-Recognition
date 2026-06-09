import streamlit as st
import numpy as np
import librosa
import joblib

from tensorflow.keras.models import load_model

 
# ======================
# Load Saved Objects
# ======================

model = load_model("speech_emotion_nn.h5")

scaler = joblib.load("scaler.pkl")

encoder = joblib.load("encoder.pkl")

# ======================
# Feature Extraction
# ======================

def extract_features(file_path):

    audio, sr = librosa.load(file_path)

    # MFCC
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfccs = np.mean(mfccs.T, axis=0)

    # ZCR
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)
    )

    # RMS
    rms = np.mean(
        librosa.feature.rms(y=audio)
    )

    # Spectral Centroid
    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    features = np.hstack([
        mfccs,
        zcr,
        rms,
        spectral_centroid
    ])

    return features

# ======================
# Streamlit UI
# ======================

st.title("🎙 Speech Emotion Recognition")

st.write(
    "Upload a WAV audio file and predict the emotion."
)

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

# ======================
# Prediction
# ======================

if uploaded_file:

    st.write("File uploaded successfully")

    try:

        features = extract_features(uploaded_file)

        st.write("Features extracted")

        features = features.reshape(1, -1)

        features = scaler.transform(features)

        st.write("Features scaled")

        prediction = model.predict(features)

        st.write("Prediction completed")

        predicted_index = np.argmax(prediction)

        emotion = encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = prediction[0][predicted_index] * 100

        st.success(
            f"Predicted Emotion: {emotion}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )

    except Exception as e:

        st.error(f"Error: {str(e)}")