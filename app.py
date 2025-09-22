import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset (deployment-safe)
@st.cache_data
def load_data():
    # Path relative to this app.py file
    url = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
    data = pd.read_csv(url)
    return data

data = load_data()

# Title
st.title("🌱 Crop Recommendation System")

# Split Data
X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Sidebar for User Input
st.sidebar.header("Enter Soil & Weather Conditions 🌦️")
N = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0, max_value=50, value=25)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
ph = st.sidebar.number_input("pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0, max_value=500, value=100)

# Prediction
if st.sidebar.button("Recommend Crop"):
    sample = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                          columns=['N','P','K','temperature','humidity','ph','rainfall'])
    prediction = model.predict(sample)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")

# Model Evaluation Section
if st.checkbox("Show Model Evaluation"):
    # Predictions on test data
    y_test_pred = model.predict(X_test)

    # Testing Accuracy as Percentage
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    st.write(f"📊 **Testing Accuracy:** {test_acc:.2f}%")

    # Classification Report
    st.text("Classification Report (Test Data):")
    st.text(classification_report(y_test, y_test_pred))

    # Confusion Matrix Heatmap
    st.write("Confusion Matrix (Test Data):")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(confusion_matrix(y_test, y_test_pred),
                annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_,
                yticklabels=model.classes_, ax=ax)
    st.pyplot(fig)