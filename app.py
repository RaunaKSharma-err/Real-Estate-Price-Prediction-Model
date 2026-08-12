import streamlit as st
import numpy as np 
import joblib

scaler = joblib.load("Scalar.pkl")
model = joblib.load("model.pkl")

st.title("Real Estate Price Prediction App")

st.divider()

bed = st.number_input("Enter the number of bedrooms",value=1,step=1)
bath = st.number_input("Enter the number of bathrooms",value= 1,step=1)
house_size = st.number_input("Enter the size of the Property",value=1000,step=50)

X = [bed,bath,house_size]

predictionButton = st.button("Predict!")

st.divider()

if predictionButton:
    
    st.balloons()
    x1 = np.array(X)
    x_array = scaler.transform([x1])
    predictions = model.predict(x_array)[0]
    st.write(f"The prediction is {predictions:.2f}")

else:
    "Please use the button for prediction"