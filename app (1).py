# app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re

# ========== FUNGSI PREPROCESSING SEDERHANA ==========
def simple_preprocess(text: str) -> str:
    # Sesuaikan dengan preprocessing saat membuat Hasil_Preprocessing_Data.csv
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', ' ', text)  # buang url, mention, hashtag
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)           # buang simbol
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========== LOAD MODEL & PREPROCESSOR ==========
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

# ========== CONFIG PAGE ==========
st.set_page_config(
    page_title="SVM Sentiment Analysis",
    layout="wide"
)

st.title("💬 Sentiment Analysis dengan SVM (TF‑IDF)")
st.write(
    "Masukkan teks bahasa Indonesia untuk diklasifikasikan menjadi "
    "**Positif** atau **Negatif**.\n\n"
    "Model: SVM (RBF) • Fitur: TF‑IDF n‑gram • Output: probabilitas per kelas."
)

# ========== SIDEBAR ==========
st.sidebar.header("ℹ️ Informasi Aplikasi")
st.sidebar.markdown(
    """
- **Model**: Support Vector Machine (SVM) dengan kernel RBF  
- **Fitur**: TF‑IDF (unigram–bigram)  
- **Input**: Teks komentar/ulasan berbahasa Indonesia  
- **Tips**: Gunakan teks yang sudah cukup bersih (tanpa emoji berlebihan).  
"""
)
st.sidebar.markdown("---")
st.sidebar.markdown("Upload file CSV di bawah untuk evaluasi batch.")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (opsional, kolom: `text`)", type=["csv"]
)

# ========== LAYOUT UTAMA ==========
tab1, tab2 = st.tabs(["🔍 Prediksi Single Teks", "📊 Evaluasi Batch (CSV)"])

# ---------- TAB 1: SINGLE TEXT ----------
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        text = st.text_area(
            "Teks ulasan / komentar:",
            height=150,
            placeholder="Contoh: Pelayanannya cepat dan sangat memuaskan..."
        )

        # DEFINISI THRESHOLD DIPINDAHKAN KE SINI
        threshold = st.slider(
            "Threshold confidence (batas keyakinan model)",
            0.1, 0.9, 0.5, 0.05
        )

        if st.button("Prediksi Sentimen", type="primary"):
            if text.strip() == "":
                st.warning("Masukkan teks terlebih dahulu.")
            else:
                # Preprocess seperti saat training
                cleaned = simple_preprocess(text)

                # Transform & predict
                X_vec = vectorizer.transform([cleaned])
                proba = model.predict_proba(X_vec)[0]
                pred  = model.predict(X_vec)[0]

                # Label & confidence
                label = le.inverse_transform([pred])[0]
                conf  = float(np.max(proba))

                is_confident = conf >= threshold

                st.subheader("Hasil Prediksi")
                st.write(f"**Teks (preprocess):** `{cleaned}`")
                st.write(
                    f"**Sentimen:** {label}  \n"
                    f"**Confidence:** {conf:.2%} "
                    f"({'di atas' if is_confident else 'di bawah'} threshold {threshold:.2f})"
                )

                st.progress(conf)

                st.write("Probabilitas:")
                st.write(f"- Positif: {proba[1]:.2%}")
                st.write(f"- Negatif: {proba[0]:.2%}")

    with col2:
        st.subheader("Contoh Cepat")
        examples = [
            "sangat bagus pelayanannya",
            "parah sekali, tidak akan beli lagi",
            "biasa saja, tidak terlalu istimewa",
            "harga mahal tapi kualitas ok",
            "murah tapi mengecewakan"
        ]

        for ex in examples:
            ex_clean = simple_preprocess(ex)
            X_ex = vectorizer.transform([ex_clean])
            proba_ex = model.predict_proba(X_ex)[0]
            pred_ex  = model.predict(X_ex)[0]
            label_ex = le.inverse_transform([pred_ex])[0]
            conf_ex  = float(np.max(proba_ex))

            st.write(f"**Teks:** {ex}")
            st.write(f"Preprocess: `{ex_clean}`")
            st.write(f"Prediksi: `{label_ex}` (conf: {conf_ex:.2%})")
            st.markdown("---")

    st.caption(
        "Catatan: preprocessing di app disederhanakan; sesuaikan dengan pipeline training Anda."
    )

# ---------- TAB 2: BATCH EVALUATION ----------
# ---------- TAB 2: BATCH EVALUATION ----------
with tab2:
    st.subheader("Evaluasi Batch dari File CSV")
    st.write(
        "Upload file `.csv` dengan minimal satu kolom bernama `text`. "
        "Jika ada kolom `label`, akan dihitung akurasi."
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("❌ CSV harus punya kolom `text`.")
        else:
            # Preprocess
            df["text_clean"] = df["text"].astype(str).apply(simple_preprocess)
            X_batch = vectorizer.transform(df["text_clean"])
            proba_batch = model.predict_proba(X_batch)
            pred_batch = model.predict(X_batch)
            label_batch = le.inverse_transform(pred_batch)

            df["pred_label"] = label_batch
            df["confidence"] = np.max(proba_batch, axis=1)

            # Tampilkan hasil
            st.dataframe(df[["text", "text_clean", "pred_label", "confidence"]].head(20))

            # METRIK RINGKASAN (TANPA SEABORN)
            st.subheader("📊 Ringkasan Hasil")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                positif = (df["pred_label"] == le.classes_[1]).sum()
                st.metric("Positif", positif)
            with col2:
                negatif = (df["pred_label"] == le.classes_[0]).sum()
                st.metric("Negatif", negatif)
            with col3:
                rata_conf = df["confidence"].mean()
                st.metric("Rata-rata Confidence", f"{rata_conf:.1%}")
            with col4:
                st.metric("Total Teks", len(df))

            # DOWNLOAD HASIL
            csv_result = df[["text", "text_clean", "pred_label", "confidence"]].to_csv(index=False)
            st.download_button("💾 Download Hasil", csv_result, "hasil_prediksi.csv")

            # Akurasi jika ada ground truth
            if "label" in df.columns:
                st.subheader("✅ Evaluasi Akurasi")
                try:
                    from sklearn.metrics import accuracy_score, classification_report
                    
                    # Map label CSV ke encoder
                    label_map = {le.classes_[i]: i for i in range(len(le.classes_))}
                    y_true_mapped = df["label"].map(label_map)
                    
                    # Hitung akurasi
                    acc = accuracy_score(y_true_mapped.dropna(), pred_batch[y_true_mapped.notna()])
                    st.success(f"**Akurasi: {acc:.2%}** ({len(y_true_mapped.dropna())} sampel)")
                    
                    # Report sederhana
                    st.info(classification_report(y_true_mapped.dropna(), pred_batch[y_true_mapped.notna()]))
                    
                except Exception as e:
                    st.warning("⚠️ Format kolom `label` tidak cocok. Cek: " + str(e))
    else:
        st.info("📁 Upload CSV di sidebar untuk analisis batch.")


st.markdown("---")
st.caption(
    "App ini menggunakan SVM + TF‑IDF. Untuk hasil terbaik, samakan preprocessing di training dan di app."
)

