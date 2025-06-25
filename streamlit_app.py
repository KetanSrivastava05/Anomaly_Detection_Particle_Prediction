import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# --- Load models and scalers ---
autoencoder = load_model("autoencoder_model.keras")
classifier = load_model("particle_classifier_model.keras")

scaler_auto = joblib.load("scaler_autoencoder.pkl")
scaler_cls = joblib.load("scaler_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# --- Constants ---
features = ['E1', 'px1', 'py1', 'pz1', 'E2', 'px2', 'py2', 'pz2', 'Q1', 'Q2']
RECON_THRESHOLD = 0.025  # Replace with your actual threshold

# --- UI ---
st.title("🔬 Particle Event Analyzer K10")
st.markdown("This app detects anomalies and predicts the parent particle of high-energy collision events.")

with st.form("event_form"):
    cols = st.columns(5)
    E1 = cols[0].number_input("E1")
    px1 = cols[1].number_input("px1")
    py1 = cols[2].number_input("py1")
    pz1 = cols[3].number_input("pz1")
    Q1  = cols[4].number_input("Q1")

    cols2 = st.columns(5)
    E2 = cols2[0].number_input("E2")
    px2 = cols2[1].number_input("px2")
    py2 = cols2[2].number_input("py2")
    pz2 = cols2[3].number_input("pz2")
    Q2  = cols2[4].number_input("Q2")

    submitted = st.form_submit_button("🔍 Analyze Event")

if submitted:
    input_vector = [E1, px1, py1, pz1, E2, px2, py2, pz2, Q1, Q2]
    df_input = pd.DataFrame([input_vector], columns=features)

    # --- Anomaly Detection ---
    X_scaled_auto = scaler_auto.transform(df_input)
    X_recon = autoencoder.predict(X_scaled_auto)
    recon_error = np.mean(np.square(X_scaled_auto - X_recon), axis=1)[0]
    is_anomaly = recon_error > RECON_THRESHOLD

    st.subheader("🔎 Anomaly Detection")
    st.write(f"**Reconstruction Error:** `{recon_error:.6f}`")
    if is_anomaly:
        st.error("❗ This event is likely an **anomaly**.")
    else:
        st.success("✅ This event appears **normal**.")

    # --- Classification ---
    X_scaled_cls = scaler_cls.transform(df_input)
    prediction_probs = classifier.predict(X_scaled_cls)
    predicted_class = np.argmax(prediction_probs)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    st.subheader("🧬 Parent Particle Prediction")
    st.write(f"**Predicted Particle:** `{predicted_label}`")
    st.write(f"**Confidence:** `{prediction_probs[0][predicted_class]*100:.2f}%`")

    # Optional: Show all class probabilities
    with st.expander("🔬 See all class probabilities"):
        prob_df = pd.DataFrame({
            "Particle": label_encoder.inverse_transform(np.arange(len(prediction_probs[0]))),
            "Confidence": prediction_probs[0]
        }).sort_values("Confidence", ascending=False).reset_index(drop=True)
        st.dataframe(prob_df.style.format({"Confidence": "{:.2%}"}))
