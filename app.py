import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score

st.set_page_config(page_title="Insurance Data Analysis", layout="wide")

st.title("🏥 Insurance Data Analysis and ML Model App")

# --- Upload or use default dataset ---
uploaded_file = st.file_uploader("Upload your insurance.csv file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("insurance.csv")

st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# --- EDA Section ---
st.header("🔍 Exploratory Data Analysis")

if st.checkbox("Show Summary Statistics"):
    st.write(df.describe())

if st.checkbox("Show BMI Distribution Plot"):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['bmi'], kde=True, ax=ax)
    st.pyplot(fig)

if st.checkbox("Show BMI Boxplot"):
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=df['bmi'], ax=ax)
    st.pyplot(fig)

# --- Outlier Removal ---
st.header("🚫 Outlier Removal (BMI)")
Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

st.write(f"Lower Bound: {lower_bound:.2f}")
st.write(f"Upper Bound: {upper_bound:.2f}")

if st.button("Remove Outliers"):
    before = len(df)
    df = df[(df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]
    after = len(df)
    st.success(f"Removed {before - after} outliers! New dataset size: {after}")

# --- Model Training Section ---
st.header("🤖 Machine Learning Model")

target_col = st.selectbox("Select Target Column", df.columns)
X = df.drop(columns=[target_col])
y = df[target_col]

# Convert categorical columns
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_choice = st.selectbox("Choose Model", ["Logistic Regression", "KNN", "Naive Bayes", "Decision Tree"])

if st.button("Train Model"):
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📈 Model Results")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Optional: visualize tree if decision tree
    if model_choice == "Decision Tree":
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_tree(model, filled=True, fontsize=8)
        st.pyplot(fig)

st.success("✅ App ready! You can now explore, clean, and model your data interactively.")
