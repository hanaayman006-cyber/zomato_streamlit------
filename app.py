import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

# ------------------------
# Paths
# ------------------------
BASE_DIR = Path(__file__).parent

# ------------------------
# Load data (sample for speed)
# ------------------------
df = pd.read_csv(BASE_DIR / "zomato_cleaned.csv")
df = df.sample(n=1000, random_state=42)  # استخدمي sample صغيرة لتجربة سريعة
# لو عايزة كل البيانات، امسحي السطر ده

# ------------------------
# Load models & scaler
# ------------------------
best_model = joblib.load(BASE_DIR / "best_model.pkl")
nn_model = load_model(BASE_DIR / "nn_model.h5")
scaler = joblib.load(BASE_DIR / "scaler.pkl")

# Features and categorical columns
X = df.drop(['rate', 'rate_cat'], axis=1)
cat_cols = df.select_dtypes(include=['object']).columns

# Scale numeric columns
X_scaled = scaler.transform(X)

# ------------------------
# Streamlit Sidebar
# ------------------------
st.sidebar.title("Zomato App")
page = st.sidebar.selectbox("Choose Page", ["Analysis", "Prediction"])

# ------------------------
# Analysis Page
# ------------------------
if page == "Analysis":
    st.title("Zomato Bangalore Restaurant Analysis")
    
    st.subheader("Top 10 Cuisines")
    st.bar_chart(df['cuisines'].value_counts().head(10))

    st.subheader("Top 10 Locations")
    st.bar_chart(df['location'].value_counts().head(10))

    st.subheader("Online Order Percentage")
    st.write(df['online_order'].value_counts(normalize=True)*100)

    st.subheader("Table Booking Percentage")
    st.write(df['book_table'].value_counts(normalize=True)*100)

    st.subheader("Top 10 Rated Restaurants")
    st.dataframe(df.sort_values(by='rate', ascending=False)[['name','rate','votes']].head(10))

# ------------------------
# Prediction Page (All Features)
# ------------------------
if page == "Prediction":
    st.title("Restaurant Rating Prediction (All Features)")
    st.write("Enter details for the restaurant:")

    input_data = {}
    for col in X.columns:
        dtype = X[col].dtype
        if np.issubdtype(dtype, np.number):
            input_data[col] = [st.number_input(col, value=float(df[col].mean()))]
        else:
            input_data[col] = [st.text_input(col, value="")]

    if st.button("Predict"):
        input_df = pd.DataFrame(input_data)

        # Encode categorical columns
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            input_df[col] = le.transform(input_df[col].astype(str))

        # Scale numeric columns
        input_scaled = scaler.transform(input_df)

        # Predictions
        pred_rf = best_model.predict(input_scaled)[0]
        pred_nn = np.argmax(nn_model.predict(input_scaled), axis=1)[0]

        rating_map = {0:"Low", 1:"Medium", 2:"High"}

        st.success(f"Random Forest/XGBoost Prediction: {rating_map[pred_rf]}")
        st.success(f"Neural Network Prediction: {rating_map[pred_nn]}")