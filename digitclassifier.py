from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow import keras
from tensorflow.keras.datasets import mnist

st.set_page_config(page_title="Digit Classifier", page_icon="✍️")
st.title("✍️ Digit Classifier")
st.write("The model will show you a digit — guess what it is!")

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "digit_classifier_model.keras"


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_data():
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    return x_test, y_test


model = load_model()
x_test, y_test = load_data()

if "index" not in st.session_state:
    st.session_state.index = np.random.randint(0, len(x_test))
if "answered" not in st.session_state:
    st.session_state.answered = False

idx = st.session_state.index
sample = x_test[idx]
true_label = int(y_test[idx])

img_display = (sample.reshape(28, 28) * 255).astype("uint8")
st.image(img_display, width=200, caption="What digit is this?")

guess = st.number_input("Your guess (0–9):", min_value=0, max_value=9, step=1)

col1, col2 = st.columns(2)

with col1:
    if st.button("Submit"):
        prediction = model.predict(sample.reshape(1, 784), verbose=0)[0]
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        st.write("---")
        st.write(f"**True label:** {true_label}")
        st.write(
            f"**Model prediction:** {predicted_digit} ({confidence:.1%} confident)"
        )

        if guess == true_label:
            st.success("You got it right!")
        else:
            st.error(f"Wrong! It was a {true_label}.")

        if predicted_digit == true_label:
            st.info("The model got it right too.")
        else:
            st.warning("The model got it wrong.")

        st.session_state.answered = True

with col2:
    if st.button("Next Digit"):
        st.session_state.index = np.random.randint(0, len(x_test))
        st.session_state.answered = False
        st.rerun()
