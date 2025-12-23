with tab2:
    st.subheader("📊 Evaluasi Batch dari File CSV")
    
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

            # ========== FIX PREDIKSI POSITIF ==========
            # FLIP jika model terbalik (negatif=0, positif=1 → flip jadi positif=0, negatif=1)
            flip_predictions = st.checkbox("🔄 Flip Prediksi (Fix Positif→Negatif)", value=True)
            if flip_predictions:
                label_batch = np.array(['positif' if x == le.classes_[0] else 'negatif' for x in label_batch])
                # Swap probabilitas
                proba_positif = proba_batch[:, 0]  # Yang tadinya negatif jadi positif
                proba_negatif = proba_batch[:, 1]  # Yang tadinya positif jadi negatif
            else:
                proba_positif = proba_batch[:, 1]
                proba_negatif = proba_batch[:, 0]

            # TAMBAH SEMUA PREDIKSI KE DF ASLI
            df["pred_label"] = label_batch
            df["confidence"] = np.maximum(proba_positif, proba_negatif)
            df["prob_positif"] = proba_positif
            df["prob_negatif"] = proba_negatif

            # TAMPILAN 20 BARIS PERTAMA
            st.dataframe(df[["text", "pred_label", "confidence", "prob_positif", "prob_negatif"]].head(20))

            # METRICS
            col1, col2, col3, col4 = st.columns(4)
            with col1: 
                st.metric("🟢 Positif", (df["pred_label"] == 'positif').sum())
            with col2: 
                st.metric("🔴 Negatif", (df["pred_label"] == 'negatif').sum())
            with col3: 
                st.metric("📊 Rata Confidence", f"{df['confidence'].mean():.1%}")
            with col4: 
                st.metric("📈 Total", len(df))

            # DOWNLOAD LENGKAP ✅
            csv_result = df.to_csv(index=False)
            st.download_button(
                "💾 Download SEMUA Kolom + Prediksi", 
                csv_result, 
                f"hasil_sentimen_{len(df)}_rows.csv"
            )

            # ========== AKURASI LENGKAP ==========
            if "label" in df.columns:
                st.subheader("🎯 Evaluasi Akurasi")
                try:
                    from sklearn.metrics import accuracy_score, classification_report
                    
                    # Map label ground truth ke model classes
                    label_map = {le.classes_[i]: i for i in range(len(le.classes_))}
                    y_true_mapped = df["label"].map(label_map)
                    valid_mask = y_true_mapped.notna()
                    
                    if valid_mask.sum() > 0:
                        # Hitung akurasi (sesuaikan dengan flip)
                        if flip_predictions:
                            y_pred_fixed = np.array([0 if x == 'positif' else 1 for x in df["pred_label"]])[valid_mask]
                        else:
                            y_pred_fixed = pred_batch[valid_mask]
                        
                        acc = accuracy_score(y_true_mapped[valid_mask], y_pred_fixed)
                        st.success(f"✅ **Akurasi: {acc:.2%}** ({valid_mask.sum()} sampel)")
                        
                        # Classification report
                        report = classification_report(
                            y_true_mapped[valid_mask], 
                            y_pred_fixed,
                            target_names=['negatif', 'positif'],
                            output_dict=False
                        )
                        st.code(report)
                    else:
                        st.warning("⚠️ Tidak ada label yang match dengan model")
                        
                except Exception as e:
                    st.error(f"❌ Error akurasi: {str(e)}")
    else:
        st.info("📁 **Upload CSV di sidebar** 👈")






