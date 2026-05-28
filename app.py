import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
from tensorflow.keras.regularizers import l2

# 모델 구조 직접 재생성
model = Sequential([
    GaussianNoise(0.1, input_shape=(5,)),
    Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# 기존 keras 파일에서 가중치만 로드
saved_model = tf.keras.models.load_model("refined_model.keras", compile=False)
model.set_weights(saved_model.get_weights())

scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("비알코올성 지방간 위험 예측 웹")
st.write("건강검진 정보를 입력하면 비알코올성 지방간 위험도를 예측합니다.")

age = st.number_input("나이", min_value=0, max_value=120, value=30)

sex_text = st.selectbox("성별", ["남성", "여성"])
sex = 1 if sex_text == "남성" else 2

HE_alt = st.number_input("ALT 수치", min_value=0.0, value=25.0)
HE_ast = st.number_input("AST 수치", min_value=0.0, value=25.0)

diabetes_text = st.selectbox("당뇨 여부", ["없음", "있음"])
DE1_pr = 1 if diabetes_text == "있음" else 0

if st.button("예측하기"):
    input_data = np.array([[age, sex, HE_alt, HE_ast, DE1_pr]])
    input_scaled = scaler.transform(input_data)

    prediction = float(model.predict(input_scaled).reshape(-1)[0])

    st.subheader("예측 결과")
    st.write(f"비알코올성 지방간 위험도: {prediction * 100:.1f}%")

    if prediction >= 0.5:
        st.error("고위험군으로 예측됩니다.")
    else:
        st.success("저위험군으로 예측됩니다.")

    st.caption("본 결과는 학습 모델 기반 예측이며 실제 의학적 진단을 대체할 수 없습니다.")