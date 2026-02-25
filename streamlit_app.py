import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb
st.title("Welcome to Energy Predictor Application")
st.write("Lets predict thev appliance energy consumption")
model = jb.load(r"energy_model.pkl")
#take the input
temp = st.number_input("Enter the temperature value: ", min_value = 0.0, max_value = 46.0, value = 5.0)

if st.button("Predict Energy Consumption"):
    new_data = np.array([[temp]])
    prediction = model.predict(new_data)
    st.write(f"The predicted energy for {temp} degree temperature is: {prediction[0]:.2f} Kwh")
    st.write("Thank You.....!")