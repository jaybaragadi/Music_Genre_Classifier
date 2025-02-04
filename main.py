import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.image import resize

# Load your pre-trained model
model = tf.keras.models.load_model("Trained_model.h5")

# Define the music genres
classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock']


# Function to load and preprocess audio data
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)


# Function to predict the genre of the music
def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]


# Streamlit app
st.title("Music Genre Classification")
st.write("Upload a song to classify its genre")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio_file.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Check Genre"):
        # Load and preprocess the audio file
        X_test = load_and_preprocess_data("temp_audio_file.wav")

        # Model prediction
        c_index = model_prediction(X_test)
        genre = classes[c_index]

        # Display the results
        st.write(f"Predicted Music Genre: **{genre}**")

        # Play the audio file
        audio_bytes = open("temp_audio_file.wav", "rb").read()
        st.audio(audio_bytes, format="audio/wav")

        # Option to check another song or close the app
        if st.button("Check Another Song"):
            st.experimental_rerun()
        if st.button("Close App"):
            st.markdown("# THANK YOU!")
            st.stop()
