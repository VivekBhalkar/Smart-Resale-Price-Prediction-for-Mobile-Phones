import streamlit as st
import pandas as pd
import joblib


model = joblib.load("model.pkl")
model_columns = joblib.load("columns.pkl")
scale_factor = joblib.load("scale.pkl")

st.title("📱 Used Phone Price Predictor")
st.write("Enter phone details below:")


device_brand = st.selectbox("Device Brand", ["Apple", "Samsung", "Xiaomi", "Realme", "OnePlus", "Others"])
os = st.selectbox("Operating System", ["Android", "iOS", "Others"])

screen_size = st.number_input("Screen Size (cm)", min_value=10.0, max_value=20.0)
four_g = st.selectbox("4G Available?", ["No", "Yes"])
five_g = st.selectbox("5G Available?", ["No", "Yes"])

front_camera_mp = st.number_input("Front Camera (MP)", min_value=0)
back_camera_mp = st.number_input("Back Camera (MP)", min_value=0)

internal_memory = st.number_input("Internal Memory (GB)", min_value=8)
ram = st.number_input("RAM (GB)", min_value=1)

battery = st.number_input("Battery (mAh)", min_value=1000)
weight = st.number_input("Weight (grams)", min_value=100)

release_year = st.number_input("Release Year", min_value=2000, max_value=2025)
days_used = st.number_input("Days Used", min_value=0)

new_price = st.number_input("New Phone Price (₹)", min_value=1000)


if st.button("Predict Price"):


    normalized_new_price = new_price / scale_factor


    input_dict = {
        "device_brand": device_brand,
        "os": os,
        "screen_size": screen_size,
        "4g": four_g,
        "5g": five_g,
        "front_camera_mp": front_camera_mp,
        "back_camera_mp": back_camera_mp,
        "internal_memory": internal_memory,
        "ram": ram,
        "battery": battery,
        "weight": weight,
        "release_year": release_year,
        "days_used": days_used,
        "normalized_new_price": normalized_new_price
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # One-hot encoding convert data into numeric form (binary)
    input_df = pd.get_dummies(input_df)

    # Match columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]

    
    predicted_price = prediction * scale_factor

    # Output
    st.success(f"💰 Predicted Used Price: ₹ {predicted_price:,.0f}")