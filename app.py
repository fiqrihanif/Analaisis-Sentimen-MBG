import streamlit as st
import sys
import os
# Menambahkan direktori 'utils' ke sys.path agar modul dapat diimpor

# Impor fungsi-fungsi dari file lokal yang telah dibuat
import pandas as pd
from utils.loaders import load_all_models, load_sastrawi
from utils.preprocessing import preprocess_input
from utils.metrics import get_metrics_data

# --- 1. SETTING HALAMAN UI ---
st.set_page_config(page_title="Perbandingan Model MBG", layout="wide")

# --- 2. MEMUAT RESOURCE (Backend) ---
model_lr, model_nb, tfidf = load_all_models()
stemmer, stopword = load_sastrawi()
df_metrics = get_metrics_data()

# --- 3. ANTARMUKA UTAMA (UI) ---
st.title("Analisis Sentimen Program Makan Bergizi Gratis (MBG)🍽️")
st.write("Aplikasi ini membandingkan kinerja model **Logistic Regression** dan **Naive Bayes** dalam mengklasifikasikan opini masyarakat.")

# --- SIDEBAR UI ---
st.sidebar.header("⚙️ Pengaturan Model")
selected_model_name = st.sidebar.selectbox(
    "Pilih Model Klasifikasi:",
    ["Logistic Regression", "Naive Bayes"]
)

# Tentukan model aktif dan metrik berdasarkan pilihan di UI
if selected_model_name == "Logistic Regression":
    active_model = model_lr
    model_info = df_metrics[df_metrics["Model"] == "Logistic Regression"]
else:
    active_model = model_nb
    model_info = df_metrics[df_metrics["Model"] == "Naive Bayes"]

st.sidebar.subheader(f"📊 Performa {selected_model_name}")
st.sidebar.metric("Accuracy", f"{model_info['Accuracy'].values[0]*100:.2f}%")
st.sidebar.metric("F1-Score", f"{model_info['F1-Score'].values[0]*100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.write("**Tabel Perbandingan Semua Model:**")
st.sidebar.dataframe(df_metrics.set_index("Model"))

# --- MAIN PANEL UI ---
st.subheader(r"💬 Pengujian Kalimat Baru")
user_input = st.text_area("Masukkan tweet atau komentar seputar program MBG di sini:", height=100)

if st.button("Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("Kolom teks tidak boleh kosong!")
    else:
        with st.spinner('Memproses teks dan melakukan prediksi...'):
            # Eksekusi fungsi preprocessing dari preprocessing.py
            cleaned_text = preprocess_input(user_input, stemmer, stopword)
            
            # Transformasi & Prediksi
            text_vector = tfidf.transform([cleaned_text])
            prediction = active_model.predict(text_vector)[0]
            
            # Tampilkan Hasil Utama di UI
            st.markdown(f"### Hasil Prediksi menggunakan **{selected_model_name}**:")
            
            if prediction == 'positif':
                st.success("Sentimen: **POSITIF** 🟢")
            elif prediction == 'negatif':
                st.error("Sentimen: **NEGATIF** 🔴")
            else:
                st.info("Sentimen: **NETRAL** ⚪")
            
            # UI Expand untuk melihat hasil di balik layar
            st.markdown("---")
            with st.expander("Lihat Detail Pemrosesan Teks & Bobot Kata"):
                st.write(f"**Teks Asli:** {user_input}")
                st.write(f"**Hasil Preprocessing:** `{cleaned_text}`")
                
                # 1. Mendapatkan kata-kata dari kalimat yang sudah di-preprocess (tokens)
                tokens = cleaned_text.split()
                
                # 2. Mendapatkan daftar seluruh fitur dari TF-IDF
                feature_names = tfidf.get_feature_names_out()
                
                # 3. Mendapatkan koefisien model (sesuai pilihan model)
                # Catatan: Jika memilih Naive Bayes, NB tidak punya .coef_ yang mudah diinterpretasi seperti LR
                # Maka visualisasi ini paling efektif jika model yang dipilih adalah Logistic Regression
                if selected_model_name == "Logistic Regression":
                    st.write("### Kontribusi Kata terhadap Kelas Prediksi")
                    
                    data_bobot = []
                    for token in tokens:
                        if token in feature_names:
                            # Cari indeks kata dalam fitur
                            idx = list(feature_names).index(token)
                            # Ambil bobot untuk semua kelas (negatif, netral, positif)
                            bobot_neg = active_model.coef_[0][idx]
                            bobot_net = active_model.coef_[1][idx]
                            bobot_pos = active_model.coef_[2][idx]
                            
                            data_bobot.append({
                                "Kata": token,
                                "Negatif": round(bobot_neg, 3),
                                "Netral": round(bobot_net, 3),
                                "Positif": round(bobot_pos, 3)
                            })
                    
                    if data_bobot:
                        df_bobot = pd.DataFrame(data_bobot)
                        st.table(df_bobot)
                    else:
                        st.write("Kata-kata dalam kalimat tidak ditemukan dalam fitur model.")
                else:
                    st.info("Visualisasi bobot kata saat ini dioptimalkan untuk Logistic Regression.")