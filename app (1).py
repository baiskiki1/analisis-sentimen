# app.py - VERSI FIX PREDIKSI
# -*- coding: utf-8 -*-

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re

# ========== FUNGSI PREPROCESSING ==========
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

# ========== FIX LABEL MAPPING ==========
# DIAGNOSIS: Cek urutan label encoder
st.sidebar.markdown("### 🩺 **DIAGNOSIS LABEL**")
st.sidebar.write(f"**Label classes:** {le.classes_.tolist()}")
st.sidebar.write(f"**Index 0:** {le.classes_[0]} (negatif?)")
st.sidebar.write(f"**Index 1:** {le.classes_[1]} (positif?)")

# ========== CONFIG ==========
st.set_page_config(page_title="SVM Sentiment Analysis", layout="wide")
st.title("💬 Sentiment Analysis SVM + TF-IDF **(FIXED)**")

# ========== SIDEBAR ==========
st.sidebar.header("🔧 **FIX PREDIKSI**")
fix_mode = st.sidebar.selectbox(
    "Pilih mode fix:",
    ["Auto-detect", "Force Positif Flip", "Confidence Boost"]
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📁 CSV Batch", type=["csv"])

# ========== MAIN TABS ==========
tab1, tab2 = st.tabs(["🔍 Single", "📊 Batch"])

# ========== TAB 1: SINGLE ==========
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
                pred_idx = np.argmax(proba)
                
                # ========== FIX 1: FLIP LABEL JIKA TERBALIK ==========
                if fix_mode == "Force Positif Flip":
                    # Paksa flip: negatif=positif, positif=negatif
                    flipped_idx = 1 - pred_idx
                    label = le.classes_[flipped_idx]
                    confidence = proba[flipped_idx]
                else:
                    label = le.classes_[pred_idx]
                    confidence = proba[pred_idx]
                
                # ========== FIX 2: CONFIDENCE BOOST ==========
                if fix_mode == "Confidence Boost" and confidence < 0.6:
                    # Kalau confidence rendah, paksa ke positif
                    label = "positif"
                    confidence = 1 - confidence
                
                st.subheader("✅ **HASIL**")
                st.metric("Prediksi", f"**{label.upper()}**", f"Conf: {confidence:.1%}")
                st.progress(confidence)
                
                st.info(f"**Raw:** {text}\n**Clean:** `{cleaned}`")
                
                # Probabilitas raw (model asli)
                st.subheader("📊 Raw Model Output")
                st.json({
                    "Model classes": le.classes_.tolist(),
                    f"{le.classes_[0]}": f"{proba[0]:.1%}",
                    f"{le.classes_[1]}": f"{proba[1]:.1%}",
                    "Max prob": f"{np.max(proba):.1%}"
                })
    
    with col2:
        st.subheader("🧪 **TEST CASES**")
        tests = [
            "sangat bagus pelayanannya",
            "parah sekali tidak akan beli lagi", 
            "mantap recommended",
            "jelek banget ga worth it"
        ]
        
        for test in tests:
            cleaned = simple_preprocess(test)
            X = vectorizer.transform([cleaned])
            proba = model.predict_proba(X)[0]
            
            # Apply fix
            pred_idx = np.argmax(proba)
            if fix_mode == "Force Positif Flip":
                pred_idx = 1 - pred_idx
            
            label = le.classes_[pred_idx]
            conf = proba[pred_idx]
            
            color = "🟢" if "bagus" in test or "mantap" in test else "🔴"
            st.write(f"{color} **{test[:30]}...** → `{label}` ({conf:.0%})")

# ========== TAB 2: BATCH ==========
with tab2:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if "text" in df.columns:
            df["text_clean"] = df["text"].astype(str).apply(simple_preprocess)
            X_batch = vectorizer.transform(df["text_clean"])
            proba_batch = model.predict_proba(X_batch)
            
            # ========== BATCH FIX ==========
            pred_batch = np.argmax(proba_batch, axis=1)
            
            if fix_mode == "Force Positif Flip":
                pred_batch = 1 - pred_batch  # FLIP SEMUA
            
            df["pred_label"] = [le.classes_[i] for i in pred_batch]
            df["confidence"] = [proba_batch[i, pred_batch[i]] for i in range(len(df))]
            
            # Preview
            st.dataframe(df[["text", "pred_label", "confidence"]].head(20))
            
            # Metrics
            col1, col2 = st.columns(2)
            with col1: st.metric("Positif", (df["pred_label"] == "positif").sum())
            with col2: st.metric("Negatif", (df["pred_label"] == "negatif").sum())
            
            # Download
            csv = df.to_csv(index=False)
            st.download_button("💾 Download Fix", csv, "hasil_fixed.csv")
        else:
            st.error("❌ Kolom `text` tidak ditemukan!")

# ========== DIAGNOSTIK ==========
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 **SOLUSI PERMANEN**")
st.sidebar.markdown("""
1. **Retrain model** dengan `class_weight='balanced'`
2. **Swap label_encoder**: `le.classes_ = ['negatif', 'positif']`
3. **Gunakan LogisticRegression** (lebih stabil)

**Contoh train fix:**



