"""Streamlit demo for the radiological triage system.

Launch: streamlit run app/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from app.model_loader import get_classifier, get_vae, get_vae_threshold, DEVICE
from src.utils.metrics import PATHOLOGY_NAMES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="Chest X-Ray Triage", layout="wide")
st.title("Chest X-Ray Radiological Triage System")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox("Classifier", ["CNN (scratch)", "DenseNet121", "ViT"])
    show_confidence = st.checkbox("Show confidence values", value=True)
    st.markdown("---")
    st.markdown("**Anomaly detection:** VAE reconstruction error")

uploaded = st.file_uploader("Upload a chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded X-ray", width=300)

    # Preprocess for classifier
    img_size = 224 if model_choice == "ViT" else 128
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img_tensor = tf(image).unsqueeze(0).to(DEVICE)

    # Preprocess grayscale for VAE
    tf_gray = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img_gray = tf_gray(image).unsqueeze(0).to(DEVICE)

    with st.spinner("Running inference..."):
        # Classification
        classifier = get_classifier(model_choice)
        with torch.no_grad():
            logits = classifier(img_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Anomaly detection
        vae = get_vae()
        score = vae.anomaly_score(img_gray).item()
        threshold = get_vae_threshold()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Pathology Predictions ({model_choice})")
        import pandas as pd
        df = pd.DataFrame({"Pathology": PATHOLOGY_NAMES, "Probability": probs})
        df = df.sort_values("Probability", ascending=False)
        st.bar_chart(df.set_index("Pathology")["Probability"])
        if show_confidence:
            st.dataframe(df.style.format({"Probability": "{:.3f}"}), height=420)

    with col2:
        st.subheader("Anomaly Score (VAE)")
        st.metric("Reconstruction Error", f"{score:.5f}",
                  delta=f"threshold={threshold:.5f}")
        if score > threshold * 1.5:
            st.error("HIGH anomaly — strongly atypical image")
        elif score > threshold:
            st.warning("MODERATE anomaly — review recommended")
        else:
            st.success("LOW anomaly — within normal distribution")

        st.progress(min(score / (threshold * 2), 1.0),
                    text=f"Anomaly level: {score / threshold * 100:.0f}% of threshold")
