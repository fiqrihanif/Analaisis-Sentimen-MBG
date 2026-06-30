import streamlit as st
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

@st.cache_resource
def load_all_models():
    """Memuat file model dan TF-IDF Vectorizer"""
    model_lr = joblib.load('D:\code\Analaisis Sentimen MBG\model\model_lr_mbg.pkl')
    model_nb = joblib.load('D:\code\Analaisis Sentimen MBG\model\model_nb_mbg.pkl')
    vectorizer = joblib.load(r'D:\code\Analaisis Sentimen MBG\model\tfidf_vectorizer.pkl')
    return model_lr, model_nb, vectorizer

@st.cache_resource
def load_sastrawi():
    """Memuat alat Stemming dan himpunan Stopwords dari Sastrawi"""
    stemmer = StemmerFactory().create_stemmer()
    stop_factory = StopWordRemoverFactory()
    # Mengambil daftar stopword Sastrawi dalam bentuk set()
    sastrawi_stopwords = set(stop_factory.get_stop_words())
    return stemmer, sastrawi_stopwords