
python
Copy
Edit
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------
# Load Data & Models
# ------------------------
@st.cache_data
def load_data(nrows=1000):
    df = pd.read_csv("zomato_cleaned.csv")
    if nrows:
        df = df.sample(n=nrows, random_state=42)  # أخد sample من الداتا
    return df

@st.cache_data
def load_models():
    best_model = joblib.load("best_model.pkl")
    nn_model = load_model("nn_model.h5")
    scaler = joblib.load("scaler.pkl")
    return best_model, nn_model, scaler

df = load_data()
best_model, nn_model, scaler = load_models()

# Features
X = df.drop(['rate', 'rate_cat'], axis=1)
cat_cols = X.select_dtypes(include=['object']).columns

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
# Prediction Page
# ------------------------
if page == "Prediction":
    st.title("Restaurant Rating Prediction (Sampled Data)")
    st.write("Enter restaurant details:")

    input_data = {}
    for col in X.columns:
        if col in cat_cols:
            input_data[col] = st.selectbox(col, df[col].unique())
        else:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        # Encode categorical columns using original data
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            input_df[col] = le.transform(input_df[col].astype(str))

        # Scale numeric columns
        numeric_cols = [c for c in input_df.columns if c not in cat_cols]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Predictions
        pred_rf = best_model.predict(input_df)[0]
        pred_nn = np.argmax(nn_model.predict(input_df), axis=1)[0]

        rating_map = {0:"Low", 1:"Medium", 2:"High"}
        st.success(f"Random Forest/XGBoost Prediction: {rating_map[pred_rf]}")
        st.success(f"Neural Network Prediction: {rating_map[pred_nn]}")