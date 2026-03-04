import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras

st.set_page_config(page_title="Digit Classifier", page_icon="✍️")
st.title("Handwritten Digit Classifier")
st.write("Upload an image of a single handwritten digit (0-9) to get a prediction.")

ROOT_DIR = Path(__file__).resolve().parent
POSSIBLE_MODEL_FILES = [
    ROOT_DIR / "digit_classifier_model.keras",
    ROOT_DIR / "digit_classifiermodel.keras",
]


def get_model_path() -> Path:
    for model_path in POSSIBLE_MODEL_FILES:
        if model_path.exists():
            return model_path
    raise FileNotFoundError(import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
        "Could not find digit classifier model. Expected one of: "
        + ", ".join(str(p.name) for p in POSSIBLE_MODEL_FILES)
    )


@st.cache_resource
def load_model():
    model_path = get_model_path()
    return keras.models.load_model(model_path), model_path


def preprocess_image(uploaded_file, target_height: int, target_width: int, channels: int):
    image = Image.open(uploaded_file)

    if channels == 1:
        image = ImageOps.grayscale(image)
    else:
        image = image.convert("RGB")

    image = image.resize((target_width, target_height))

    img_array = np.array(image, dtype=np.float32)

    if channels == 1:
        if img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)
    else:
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return image, img_array


try:
    model, model_path = load_model()
    st.success(f"Loaded model: {model_path.name}")
except Exception as error:
    st.error(f"Model could not be loaded: {error}")
    st.stop()

input_shape = model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]

_, height, width, channels = input_shape

uploaded_file = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    preview_img, processed = preprocess_image(
        uploaded_file,
        target_height=height,
        target_width=width,
        channels=channels,
    )

    st.image(preview_img, caption="Processed input image", width=200)

    if st.button("Predict Digit"):
        prediction = model.predict(processed, verbose=0)

        if prediction.ndim == 2 and prediction.shape[1] > 1:
            predicted_digit = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0]))
            st.subheader(f"Prediction: {predicted_digit}")
            st.write(f"Confidence: {confidence:.2%}")

            st.write("Class probabilities:")
            for i, prob in enumerate(prediction[0]):
                st.write(f"{i}: {prob:.2%}")
        else:
            predicted_value = float(prediction.squeeze())
            st.subheader(f"Prediction output: {predicted_value:.4f}")
