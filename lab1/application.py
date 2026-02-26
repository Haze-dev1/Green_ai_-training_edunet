import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb

# Title
st.title("Welcome to Solar Power Prediction Application")
st.write("Lets predict the Solar Power")

# Load model
model = jb.load(r"solar_power_prediction_model.pkl")

# Take input from Streamlit UI
temp = st.number_input("Enter the Temperature (°C):", min_value=0.0, max_value=100.0, step=0.1)

hum = st.number_input("Enter the Humidity (%):", min_value=0.0, max_value=100.0, step=0.1)

solar = st.number_input("Enter the Solar Irradiance (W/m²):", min_value=0.0, step=0.1)

wind = st.number_input("Enter the Wind Speed (m/s):", min_value=0.0, step=0.1)


# Prediction button
if st.button("Predict Solar Power"):

    new_data = np.array([[temp, hum, solar, wind]])

    prediction = model.predict(new_data)

    st.success(f"The predicted solar energy is: {prediction[0]:.2f} watts")

    st.write("Thank You.....!")
