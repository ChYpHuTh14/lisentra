import streamlit as st
import librosa
import torch
import numpy as np
import pickle
import os
import soundfile as sf
from audio_recorder_streamlit import audio_recorder

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

@st.cache_resource
def load_model_and_processor(model_path="wav2vec2", le_path="le.pkl"):
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        st.success("Model, Processor, dan Label Encoder berhasil dimuat!")
        return processor, model, le, device
    except Exception as e:
        st.error(f"**Gagal memuat model atau processor!** Pastikan file 'wav2vec2' dan 'le.pkl' ada. Error: {e}")
        st.stop()

def predict_emotion(audio_data, sampling_rate, processor, model, le, device):
    audio_data = audio_data.astype(np.float32)
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        if attention_mask is not None:
            logits = model(input_values, attention_mask=attention_mask).logits
        else:
            logits = model(input_values).logits

    predicted_id = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    emotion = le.inverse_transform([predicted_id])[0]
    return emotion

st.set_page_config(
    page_title="Speech Emotion Recognition (SER) App",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="🎤"
)

st.title("🗣️ Aplikasi Prediksi Emosi Suara")
st.markdown("Unggah file audio (`.wav`) Anda atau **rekam suara langsung** untuk memprediksi emosi.")

processor, model, le, device = load_model_and_processor()

tab1, tab2 = st.tabs(["⬆️ Unggah File Audio", "🎙️ Rekam Langsung"])

with tab1:
    st.subheader("Unggah File Audio (.wav)")
    uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        temp_audio_path_upload = "temp_uploaded_audio.wav"
        with open(temp_audio_path_upload, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("---")
        st.info("Menganalisis emosi dari file yang diunggah...")
        
        try:
            audio_data, sampling_rate = librosa.load(temp_audio_path_upload, sr=16000)
            
            if audio_data.shape[0] == 0:
                st.warning("⚠️ **File audio kosong atau tidak dapat diproses.** Silakan coba file lain.")
            else:
                predicted_emotion = predict_emotion(audio_data, sampling_rate, processor, model, le, device)
                st.write("---")
                st.success(f"### 🎉 **Emosi yang Diprediksi:** {predicted_emotion.upper()} 🎉")

        except Exception as e:
            st.error(f"❌ **Terjadi kesalahan saat memproses audio Anda!** Pastikan file .wav valid. Error: {e}")
        
        if os.path.exists(temp_audio_path_upload):
            os.remove(temp_audio_path_upload)

with tab2:
    st.subheader("Rekam Suara Langsung")
    st.write("klik tombol microphone untuk mulai perekaman selama 3 detik")

    wav_audio_data = audio_recorder(
        text="Klik untuk Merekam", 

        energy_threshold=(-1.0, 1.0), 
        pause_threshold=3.0,
        sample_rate=16000, 
        key="audio_recorder_ser"
    )

    if wav_audio_data is not None:

        temp_audio_path_record = "temp_recorded_audio.wav"
        
        try:
            with open(temp_audio_path_record, "wb") as f:
                f.write(wav_audio_data)

            st.write("---")
            st.info("Menganalisis emosi dari rekaman Anda...")
            
            audio_data_record, sampling_rate_record = librosa.load(temp_audio_path_record, sr=16000)

            if audio_data_record.shape[0] == 0:
                st.warning("⚠️ **Rekaman kosong atau tidak dapat diproses.** Silakan coba rekam lagi.")
            else:
                predicted_emotion_record = predict_emotion(audio_data_record, sampling_rate_record, processor, model, le, device)
                st.write("---")
                st.success(f"### 🎉 **Emosi yang Diprediksi:** {predicted_emotion_record.upper()} 🎉")

        except Exception as e:
            st.error(f"❌ **Terjadi kesalahan saat memproses rekaman Anda!** Error: {e}")
        
        if os.path.exists(temp_audio_path_record):
            os.remove(temp_audio_path_record)

st.markdown("---")
st.markdown("Model: Wav2Vec2")