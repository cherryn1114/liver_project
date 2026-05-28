import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
# 페이지 설정
st.set_page_config(
    page_title="비알코올성 지방간 예측 AI",
    page_icon="🩺",
    layout="centered"
)
# CSS 스타일
st.markdown("""
<style>
.main {
    background-color: #F7F9FC;
}
.stButton > button {
    background-color: #1E88E5;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
}
.stButton > button:hover {
    background-color: #1565C0;
    color: white;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)
# 모델 및 scaler 불러오기
model = tf.keras.models.load_model(
    "compatible_model.h5",
    compile=False
)
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
# 제목
st.markdown(
    """
    <h1 style='text-align: center; color: #1E88E5;'>
    비알코올성 지방간 위험 예측 AI
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style='text-align: center; font-size:18px;'>
    건강검진 정보를 입력하면 지방간 위험도를 예측합니다.
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")
st.write("")
# 사이드바
st.sidebar.title("프로젝트 정보")
st.sidebar.info(
    """
    본 웹은 인공신경망(ANN)을 이용한
    비알코올성 지방간 위험 예측 시스템입니다.
    입력 변수:
    - 나이
    - 성별
    - ALT
    - AST
    - 당뇨 여부
    """
)
# 입력 UI
st.markdown("## 건강검진 정보 입력")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        "나이",
        min_value=0,
        max_value=120,
        value=30
    )
    HE_alt = st.number_input(
        "ALT 수치",
        min_value=0.0,
        value=25.0
    )
    sex_text = st.selectbox(
        "성별",
        ["남성", "여성"]
    )
with col2:
    HE_ast = st.number_input(
        "AST 수치",
        min_value=0.0,
        value=25.0
    )
    diabetes_text = st.selectbox(
        "당뇨 여부",
        ["없음", "있음"]
    )
# 변수 변환
sex = 1 if sex_text == "남성" else 2
DE1_pr = 1 if diabetes_text == "있음" else 8
st.write("")
st.write("")
# 예측 버튼
if st.button("위험도 예측하기"):
    # 입력 데이터 생성
    input_data = np.array([
        [age, sex, HE_alt, HE_ast, DE1_pr]
    ])
    # 스케일링
    input_scaled = scaler.transform(input_data)
    # 예측
    prediction = model.predict(input_scaled)
    prediction_value = float(
        prediction.reshape(-1)[0]
    )
    risk_percent = prediction_value * 100
    st.write("")
    st.markdown("---")
    # 결과 제목
    st.markdown(
        """
        <h2 style='text-align: center; color: #1E88E5;'>
        예측 결과
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    # 위험도 바
    st.progress(int(risk_percent))
    # 위험도 수치
    st.metric(
        label="비알코올성 지방간 위험도",
        value=f"{risk_percent:.1f}%"
    )
    st.write("")
    # 위험군 판정
    if prediction_value >= 0.5:
        st.error(
            "고위험군으로 예측됩니다. "
            "정확한 진단을 위해 전문의 상담을 권장합니다."
        )
    else:
        st.success(
            "저위험군으로 예측됩니다."
        )
    # 안내문
    st.caption(
        "본 결과는 AI 기반 예측 결과이며 "
        "실제 의학적 진단을 대체하지 않습니다."
    )