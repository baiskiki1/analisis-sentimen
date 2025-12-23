with tab2:
    st.subheader("Evaluasi Batch dari File CSV")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("❌ CSV harus punya kolom `text`.")
        else:
            # Preprocess & Predict
            df["text_clean"] = df["text"].astype(str).apply(simple_preprocess)
            X_batch = vectorizer.transform(df["text_clean"])
            proba_batch = model.predict_proba(X_batch)
            pred_batch = model.predict(X_batch)
            label_batch = le.inverse_transform(pred_batch)

            # TAMBAH SEMUA PREDIKSI KE DF ASLI
            df["pred_label"] = label_batch
            df["confidence"] = np.max(proba_batch, axis=1)
            df["prob_positif"] = proba_batch[:, 1]
            df["prob_negatif"] = proba_batch[:, 0]

            # TAMPILAN 20 BARIS PERTAMA
            st.dataframe(df[["text", "pred_label", "confidence"]].head(20))

            # METRICS
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Positif", (df["pred_label"] == le.classes_[1]).sum())
            with col2: st.metric("Negatif", (df["pred_label"] == le.classes_[0]).sum())
            with col3: st.metric("Rata Confidence", f"{df['confidence'].mean():.1%}")

            # DOWNLOAD LENGKAP ✅
            csv_result = df.to_csv(index=False)
            st.download_button("💾 Download SEMUA Kolom + Prediksi", csv_result, "hasil_lengkap.csv")

            # Akurasi jika ada label
            if "label" in df.columns:
                # ... kode akurasi sama ...





