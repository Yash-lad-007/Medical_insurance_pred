import streamlit as st
import pandas as pd
import pickle

# Load your trained model
with open("Best_model (1).pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset to get column info (optional)
data = pd.read_csv("insurance.csv")

st.set_page_config(page_title="Insurance Cost Prediction", page_icon="💰", layout="centered")

st.title("💰 Insurance Cost Prediction App")
st.write("This app predicts **insurance charges** based on user details using a trained ML model.")

# Automatically detect input features from dataset (excluding target)
target_col = "charges" if "charges" in data.columns else None
features = [col for col in data.columns if col != target_col]

# Create dynamic input fields
user_input = {}
for col in features:
    if data[col].dtype == "object":
        user_input[col] = st.selectbox(f"Select {col}", sorted(data[col].unique()))
    else:
        user_input[col] = st.number_input(f"Enter {col}", min_value=float(data[col].min()), max_value=float(data[col].max()), value=float(data[col].mean()))

# Convert user input to dataframe
input_df = pd.DataFrame([user_input])

# Prediction button
if st.button("Predict Insurance Charges"):
    try:
        prediction = model.predict(input_df)
        st.success(f"💵 Estimated Insurance Cost: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Optional section: show dataset preview
with st.expander("View Sample Dataset"):
    st.dataframe(data.head())
