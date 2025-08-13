import streamlit as st
import pandas as pd
import pickle

# =====================
# تحميل البيانات (Sample فقط)
# =====================
df = pd.read_csv("zomato_cleaned.csv").sample(500, random_state=42)  # 500 صف فقط

# =====================
# تحميل النماذج والمحول
# =====================
best_model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =====================
# تحديد الأعمدة
# =====================
X = df.drop("rate", axis=1)  # غير "rate" لو العمود الهدف مختلف
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# =====================
# واجهة Streamlit
# =====================
st.title("🍽️ Zomato Restaurant Rating Prediction")

st.sidebar.header("Choose Page")
page = st.sidebar.selectbox("Select Page", ["Prediction"])

if page == "Prediction":
    st.header("Restaurant Rating Prediction (Sample Data)")

    input_data = {}
    for col in X.columns:
        if col in cat_cols:
            input_data[col] = st.selectbox(col, df[col].unique())
        else:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

        # تحويل الأعمدة النصية إلى أرقام (one-hot أو encoding)
        input_encoded = pd.get_dummies(input_df)
        df_encoded = pd.get_dummies(X)

        # جعل الأعمدة متوافقة
        input_encoded = input_encoded.reindex(columns=df_encoded.columns, fill_value=0)

        # Scaling
        input_scaled = scaler.transform(input_encoded)

        # التنبؤ
        prediction = best_model.predict(input_scaled)[0]
        st.success(f"✅ Predicted Rating: {prediction}")