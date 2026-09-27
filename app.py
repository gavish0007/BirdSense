import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import tempfile
import os
from tensorflow.keras.models import load_model

MODEL_PATH = "models/birdsense_final.keras"

class_names = [
    "common_koel",
    "common_myna",
    "common_tailorbird",
    "greater_coucal",
    "house_sparrow",
    "indian_robin",
    "red_vented_bulbul",
    "rose_ringed_parakeet"
]

model = load_model(MODEL_PATH)

def preprocess_audio(path):
    # NOTE: sr must be fixed at 22050 to exactly match the training
    # pipeline (01_dataset_exploration.ipynb / BirdSense_Final.ipynb,
    # preprocess_audio()). Using sr=None (native rate) was the #1 cause
    # of prediction drift, since every mel spectrogram used to train
    # birdsense_final.keras was computed at a fixed 22050 Hz.
    audio, sr = librosa.load(path, sr=22050, mono=True)
    audio, _ = librosa.effects.trim(audio)
    return audio, sr

def segment_audio(audio, sr, segment_duration=4):
    # NOTE: must match the notebook's segment_audio() exactly, used
    # inside process_recording() to build the training PNGs: only
    # full-length segments are kept. Any leftover tail shorter than
    # segment_length is dropped, NOT zero-padded and kept. Padding
    # and keeping it (as this function previously did) fed the model
    # an extra, partially-silent segment it never saw during training.
    segment_length = int(segment_duration * sr)
    segments = []

    for start in range(0, len(audio), segment_length):
        end = start + segment_length
        segment = audio[start:end]

        if len(segment) == segment_length:
            segments.append(segment)

    return segments

def enhance_spectrogram(mel_db):

    normalized = cv2.normalize(
        mel_db,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    normalized = normalized.astype(np.uint8)

    gaussian = cv2.GaussianBlur(
        normalized,
        (5, 5),
        0
    )

    median = cv2.medianBlur(
        normalized,
        5
    )

    equalized = cv2.equalizeHist(
        normalized
    )

    enhanced = cv2.normalize(
        equalized,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    enhanced = enhanced.astype(np.uint8)

    return normalized, gaussian, median, enhanced

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(
        image,
        (172, 128)
    )

    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(
        image,
        axis=-1
    )

    return image


def predict_bird(audio_path):

    audio, sr = preprocess_audio(audio_path)

    segments = segment_audio(
        audio,
        sr,
        segment_duration=4
    )

    if len(segments) == 0:
        # Dropping incomplete tail segments (to match training) means
        # audio shorter than 4 seconds yields zero segments. The
        # notebook never hit this because every training file was
        # already >=4s, but a live upload can be shorter.
        raise ValueError(
            "This clip is shorter than 4 seconds after trimming "
            "silence. Please upload a longer recording (at least "
            "4 seconds of audio)."
        )

    predictions = []
    first_enhanced = None

    for i, segment in enumerate(segments):

        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        normalized, gaussian, median, enhanced = enhance_spectrogram(
            mel_db
        )

        if first_enhanced is None:
            first_enhanced = enhanced

        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temp_image:

            temp_image_path = temp_image.name

        plt.imsave(
            temp_image_path,
            enhanced,
            cmap="gray"
        )

        image = load_image(temp_image_path)

        image = np.expand_dims(
            image,
            axis=0
        )

        prediction = model.predict(
            image,
            verbose=0
        )[0]

        predictions.append(prediction)

        os.remove(temp_image_path)

    predictions = np.array(predictions)

    final_prediction = predictions.mean(axis=0)

    predicted_class = np.argmax(final_prediction)

    bird = class_names[predicted_class]

    confidence = final_prediction[predicted_class]

    return bird, confidence, first_enhanced

st.set_page_config(
    page_title="BirdSense",
    page_icon="🐦",
    layout="centered"
)

st.title("🐦 BirdSense")
st.subheader("Bird Sound Classification System")

st.write(
    "Upload a bird audio recording and BirdSense "
    "will predict the bird species."
)

uploaded_file = st.file_uploader(
    "Upload bird audio",
    type=["mp3", "wav", "ogg", "flac"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Predict Bird"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        try:
            bird, confidence, spectrogram = predict_bird(
                temp_path
            )

            st.success(
                f"Predicted Bird: {bird}"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            librosa.display.specshow(
                spectrogram,
                x_axis="time",
                y_axis="mel",
                ax=ax
            )

            ax.set_title(
                "Enhanced Mel Spectrogram"
            )

            fig.colorbar(
                ax.collections[0],
                ax=ax
            )

            st.pyplot(fig)

        except ValueError as e:
            st.error(str(e))

        finally:
            os.remove(temp_path)