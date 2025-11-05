import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.title("🏥 Insurance Charges Prediction App")
st.write("Predict medical insurance costs based on user inputs or uploaded data using your trained model.")

# --- Load Trained Model ---
@st.cache_resource
def load_model():
    with open("Best_model (1).pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# --- Sidebar for user input method ---
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("How would you like to input data?", ["Manual Entry", "Upload CSV"])

# --- Manual Input Form ---
def get_user_input():
    st.subheader("🧾 Enter Details Manually")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    # Convert categorical variables to match training
    user_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Encode categorical columns similar to training
    user_data = pd.get_dummies(user_data, drop_first=True)
    return user_data

# --- Upload CSV Option ---
def load_uploaded_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())
    return data

# --- Handle user input ---
if input_method == "Manual Entry":
    user_data = get_user_input()
else:
    uploaded_file = st.file_uploader("Upload your insurance data CSV", type=["csv"])
    if uploaded_file is not None:
        user_data = load_uploaded_data(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        user_data = None

# --- Predict button ---
if st.button("🔮 Predict Insurance Charges"):
    if user_data is not None:
        # Align columns with training model input
        try:
            # The model expects specific columns, so handle missing ones
            model_features = model.feature_names_in_
            for col in model_features:
                if col not in user_data.columns:
                    user_data[col] = 0
            user_data = user_data[model_features]

            predictions = model.predict(user_data)
            st.subheader("💰 Predicted Insurance Charges:")
            for i, pred in enumerate(predictions):
                st.write(f"**Person {i+1}:** ₹{pred:,.2f}")
        except Exception as e:
            st.error(f"Error while predicting: {e}")
    else:
        st.warning("Please provide valid input data before predicting.")

st.info("💡 Tip: You can upload a CSV containing multiple entries to get batch predictions.")
