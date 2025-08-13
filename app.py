import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path

# -------- Load models and data --------
BASE_DIR = Path(__file__).parent

# Load ML model
best_model = joblib.load(BASE_DIR / 'best_model.pkl')

# Load Neural Network model
nn_model = tf.keras.models.load_model(BASE_DIR / 'nn_model.h5')

# Load Scaler
scaler = joblib.load(BASE_DIR / 'scaler.pkl')

# Load cleaned dataset for Analysis page
df = pd.read_csv(BASE_DIR / 'zomato_cleaned.csv')

# -------- Sidebar for page selection --------
page = st.sidebar.selectbox("Choose Page", ["Analysis", "Prediction"])

# -------- Analysis Page --------
if page == "Analysis":
    st.title("Zomato Data Analysis")
    
    st.subheader("First 5 rows of dataset:")
    st.dataframe(df.head())
    
    st.subheader("Top 10 Cuisines:")
    st.bar_chart(df['cuisines'].value_counts().head(10))
    
    st.subheader("Top 10 Locations:")
    st.bar_chart(df['location'].value_counts().head(10))
    
    st.subheader("Distribution of Ratings:")
    st.bar_chart(df['rate'].value_counts().sort_index())

# -------- Prediction Page --------
elif page == "Prediction":
    st.title("Restaurant Rating Prediction")
    
    # Input form
    votes = st.number_input("Votes", min_value=0, value=100)
    rate = st.number_input("Rating", min_value=0.0, max_value=5.0, value=3.0)
    avg_cost_log = st.number_input("Log Average Cost (for 2 ppl)", value=7.0)
    online_order = st.selectbox("Online Order", ["Yes","No"])
    book_table = st.selectbox("Book Table", ["Yes","No"])
    
    if st.button("Predict"):
        # Feature Engineering on input
        Votes_per_Rating = votes / rate
        High_Cost = 1 if avg_cost_log > df['avg_cost_log'].median() else 0
        Online_Order = 1 if online_order=="Yes" else 0
        Book_Table = 1 if book_table=="Yes" else 0

        # Prepare input array
        input_features = np.array([[votes, rate, avg_cost_log, Votes_per_Rating, High_Cost, Online_Order, Book_Table]])
        input_features_scaled = scaler.transform(input_features)
        
        # Predictions
        pred_ml = best_model.predict(input_features_scaled)[0]
        pred_nn = np.argmax(nn_model.predict(input_features_scaled), axis=1)[0]
        
        st.subheader("Predictions:")
        st.write(f"✅ Best ML Model Prediction: {pred_ml}")
        st.write(f"✅ Neural Network Prediction: {pred_nn}")