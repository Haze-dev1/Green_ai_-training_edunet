import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("satellite_model.keras")

# Class labels and icons
class_names = ["cloudy", "desert", "green_area", "water"]
class_icons = {"cloudy": "☁️", "desert": "🏜️", "green_area": "🌿", "water": "🌊"}
class_colors = {"cloudy": "#7f8c8d", "desert": "#e67e22", "green_area": "#27ae60", "water": "#2980b9"}

st.set_page_config(page_title="Satellite Image Classifier", layout="wide", page_icon="🛰️")

# Header
st.markdown("## 🛰️ Satellite Image Classifier")
st.markdown("Classify land types from satellite imagery using a CNN model.")
st.divider()



# Upload section
uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
st.markdown("##### Upload a satellite image")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Two-column layout: image | results
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 🖼️ Uploaded Image")
        st.image(image, use_column_width=True)
        st.caption(f"File: `{uploaded_file.name}` | Size: {image.size[0]}×{image.size[1]}px")

    # Preprocess
    img = image.resize((255, 255))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    with col2:
        st.markdown("#### 📊 Results")

        # Prediction badge
        icon = class_icons[predicted_class]
        color = class_colors[predicted_class]
        st.markdown(
            f"<div style='background:{color};padding:16px 24px;border-radius:10px;text-align:center;'>"
            f"<span style='font-size:2rem'>{icon}</span><br>"
            f"<span style='color:white;font-size:1.4rem;font-weight:bold'>{predicted_class.replace('_',' ').title()}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("")

        # Confidence metric
        st.metric(label="Confidence", value=f"{confidence:.1f}%")
        st.progress(int(confidence))

        # All class probabilities
        st.markdown("**Class Probabilities**")
        for i, name in enumerate(class_names):
            prob = prediction[0][i] * 100
            st.write(f"{class_icons[name]} {name.replace('_', ' ').title()}")
            st.progress(int(prob), text=f"{prob:.1f}%")

else:
    st.info("Upload a `.jpg`, `.png`, or `.jpeg` image above to classify it.", icon="📂")