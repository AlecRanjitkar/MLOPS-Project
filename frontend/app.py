import io

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Fashion-MNIST Classifier", layout="centered")

st.title("👕 Fashion-MNIST Classifier")
st.write("Upload an image of clothing and the model will predict the class.")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=250)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            files = {"file": uploaded.getvalue()}
            r = requests.post(API_URL, files={"file": ("image.png", uploaded.getvalue(), "image/png")})

        if r.status_code == 200:
            res = r.json()
            st.success(f"Prediction: **{res['predicted_class']}**")
            st.write(f"Confidence: `{res['confidence']:.2f}`")
        else:
            st.error(f"API Error: {r.text}")
