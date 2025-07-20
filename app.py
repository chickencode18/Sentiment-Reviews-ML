import streamlit as st
import joblib
import pandas as pd

# Load mô hình đã huấn luyện
model = joblib.load("model.pkl")

# Giao diện web
st.title("Amazon Review Sentiment Classifier")

review_summary = st.text_input("Summary:")
review_text = st.text_area("Text:")
helpfulness_numerator = st.number_input("numerator:", min_value=0)
helpfulness_denominator = st.number_input("denominator:", min_value=1)

if st.button("Dự Đoán"):
    # Xử lý dữ liệu đầu vào
    helpfulness_ratio = helpfulness_numerator / (helpfulness_denominator + 1e-5)
    all_text = review_summary.lower() + " " + review_text.lower()

    # Tạo dataframe cho model
    input_df = pd.DataFrame({
        "HelpfulnessRatio": [helpfulness_ratio],
        "All_Text": [all_text]
    })

    # Dự đoán
    prediction = model.predict(input_df)[0]

    # Hiển thị kết quả
    if prediction == 1:
        st.success("Positive ✅")
    else:
        st.error("Negative ❌")
