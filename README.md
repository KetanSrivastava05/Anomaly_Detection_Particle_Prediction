# 🧬 Particle Collision Event Analyzer

This project uses deep learning to:

- 🔍 Detect anomalous high-energy physics events using an **Autoencoder**.
- 🧪 Predict the **parent particle** responsible for a collision event using a **classification model**.

All powered by a **Streamlit web interface** for interactive usage.

---

## 🧪 Project Structure

├── streamlit_app.py # Main Streamlit interface
├── dielectron.csv # Raw input data (if applicable)
├── autoencoder_model.h5/.keras # Trained Autoencoder for anomaly detection
├── particle_classifier_model.h5/.keras # Trained classifier for parent particle prediction
├── scaler_autoencoder.pkl # Scaler used for anomaly model input
├── scaler_classifier.pkl # Scaler used for classifier input
├── label_encoder.pkl # Encoder for class labels (e.g., "Z boson", "Jpsi", etc.)
├── README.md # This file

pip install streamlit pandas numpy scikit-learn tensorflow joblib


streamlit run streamlit_app.py
