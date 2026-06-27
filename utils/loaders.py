import os
import joblib
import streamlit as st

@st.cache_resource
def load_all_models():
    # 1. Tentukan folder dasar tempat file ini (loaders.py) berada
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Cari folder 'model' yang berada satu tingkat di atas folder 'utils'
    # (Jika folder 'model' sejajar dengan folder 'utils')
    project_root = os.path.dirname(base_path)
    model_path = os.path.join(project_root, 'model')
    
    # 3. Load model menggunakan path yang sudah dinamis
    model_lr = joblib.load(os.path.join(model_path, 'model_lr_mbg.pkl'))
    model_nb = joblib.load(os.path.join(model_path, 'model_nb_mbg.pkl'))
    vectorizer = joblib.load(os.path.join(model_path, 'tfidf_vectorizer.pkl'))
    
    return model_lr, model_nb, vectorizer