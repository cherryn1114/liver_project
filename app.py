import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

모델 불러오기

model = tf.keras.models.load_model("refined_model.keras")
scaler = joblib.load(“scaler.pkl”)
features = joblib.load(“features.pkl”)

제목

st.title(“비알코올성 지방간 위험 예측 웹”)

st.write(“건강검진 정보를 입력하면 지방간 위험도를 예측합니다.”)

입력값

age = st.number_input(
“나이”,
min_value=0,
max_value=120,
value=30
)

sex_text = st.selectbox(
“성별”,
[“남성”, “여성”]
)

sex = 1 if sex_text == “남성” else 2

HE_alt = st.number_input(
“ALT 수치”,
min_value=0.0,
value=25.0
)

HE_ast = st.number_input(
“AST 수치”,
min_value=0.0,
value=25.0
)

diabetes_text = st.selectbox(
“당뇨 여부”,
[“없음”, “있음”]
)

DE1_pr = 1 if diabetes_text == “있음” else 0

예측

if st.button(“예측하기”):

input_data = np.array([
    [age, sex, HE_alt, HE_ast, DE1_pr]
])
input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0][0]
st.subheader("예측 결과")
st.write(
    f"비알코올성 지방간 위험도: {prediction * 100:.1f}%"
)
if prediction >= 0.5:
    st.error("고위험군으로 예측됩니다.")
else:
    st.success("저위험군으로 예측됩니다.")
st.caption(
    "본 결과는 학습 모델 기반 예측이며 실제 의학적 진단을 대체할 수 없습니다."
)
