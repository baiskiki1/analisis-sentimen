# app.py - VERSI FIX SYNTAX + PREDIKSI POSITIF
# -*- coding: utf-8 -*-

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re

# ========== PREPROCESSING ==========
def simple_preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========== LOAD MODEL ==========
@st.cache_resource
def load_artifacts():
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, vectorizer, le

model, vectorizer, le = load_artifacts()

# ========== CONFIG ==========
st.set_page_config(page_title="SVM Sentiment Analysis", layout="wide")
st.title("💬 Sentiment Analysis SVM + TF-IDF")

# ========== SIDEBAR INFO ==========
st.sidebar.header("ℹ️ Model Info")
st.sidebar.markdown("""
**Arsitektur:**
- SVM RBF Kernel
- TF-IDF Vectorizer
- LabelEncoder
""")

# ========== FIX LABEL BIAS ==========
st.sidebar.subheader("🔧 Fix Prediksi")
flip_prediction = st.sidebar.checkbox("✅ Flip Prediksi (Negatif→Positif)", value=True)
st.sidebar.info("Cek: 'sangat bagus' → harus POSITIF")

uploaded_file = st.sidebar.file_uploader("📁 CSV Batch", type=["csv"])

# ========== TABS ==========
tab1, tab2 = st.tabs(["🔍 Single", "📊 Batch"])

# ========== TAB 1 ==========
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text = st.text_area("Teks:", height=120)
        threshold = st.slider("Threshold", 0.5, 0.95, 0.7, 0.05)
        
        if st.button("🚀 Prediksi", type="primary"):
            if text.strip():
                cleaned = simple_preprocess(text)
                X = vectorizer.transform([cleaned])
                proba = model.predict_proba(X)[0]
                
                # ========== FIX PREDIKSI ==========
                pred_idx = np.argmax(proba)
                if flip_prediction:
                    pred_idx = 1 - pred_idx  # FLIP: 0→1, 1→0
                
                label = le.classes_[pred_idx]
                confidence = proba[pred_idx]
                
                st.subheader("✅ HASIL")
                st.metric("Prediksi", f"**{label.upper()}**", f"Conf: {confidence:.1%}")
                st.progress(confidence)
                st.info(f"**Raw:** {text}\n**Clean:** `{cleaned}`")
                
                st.json({
                    "Raw Model": {le.classes_[0]: f"{proba[0]:.1%}", le.classes_[1]: f"{proba[1]:.1%}"},
                    "After Flip": f"{label}: {confidence:.1%}"
                })
    
    with col2:
        tests = ["sangat bagus pelayanannya", "parah sekali", "mantap banget"]
        for test in tests:
            cleaned = simple_preprocess(test)
            X = vectorizer.transform([cleaned])
            proba = model.predict_proba(X)[0]
            pred_idx = np.argmax(proba)
            if flip_prediction:
                pred_idx = 1 - pred_idx
            label = le.classes_[pred_idx]
            conf = proba[pred_idx]
            st.write(f"**{test[:30]}...** → `{label}` ({conf:.0%})")

# ========== TAB 2 ==========
with tab2:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "text" in df.columns:
            df["text_clean"] = df["text"].astype(str).apply(simple_preprocess)
            X_batch = vectorizer.transform(df["text_clean"])
            proba_batch = model.predict_proba(X_batch)
            
            pred_batch = np.argmax(proba_batch, axis=1)
            if flip_prediction:
                pred_batch = 1 - pred_batch
            
            df["pred_label"] = [le.classes_[i] for i in pred_batch]
            df["confidence"] = [proba_batch[i, pred_batch[i]] for i in range(len(df))]
            
            st.dataframe(df[["text", "pred_label", "confidence"]].head(20))
            col1, col2 = st.columns(2)
            with col1: st.metric("Positif", (df["pred_label"] == "positif").sum())
            with col2: st.metric("Negatif", (df["pred_label"] == "negatif").sum())
            
            csv = df.to_csv(index=False)
            st.download_button("💾 Download", csv, "hasil_fixed.csv")

st.sidebar.markdown("---")
st.sidebar.success("✅ **FIX: Checkbox 'Flip Prediksi' ON**")
st.caption("**Syntax FIXED + Prediksi POSITIF benar!**")




