import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="비알코올성 지방간 위험 예측 AI",
    page_icon="🩺",
    layout="centered"
)
# =========================
# 스타일
# =========================
st.markdown("""
<style>
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
}
</style>
""", unsafe_allow_html=True)
# =========================
# 모델 불러오기
# =========================
model = tf.keras.models.load_model(
    "compatible_model.h5",
    compile=False
)
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
# =========================
# 제목
# =========================
st.markdown(
    """
    <h1 style='text-align:center; color:#1E88E5;'>
    비알코올성 지방간 위험 예측 AI
    </h1>
    """,
    unsafe_allow_html=True
)
st.write(
    "건강검진 정보를 입력하면 비알코올성 지방간 위험 수준을 예측합니다."
)
# =========================
# 사이드바
# =========================
st.sidebar.title("프로젝트 정보")
st.sidebar.info(
    """
    입력 변수
    • 나이
    • 성별
    • ALT
    • AST
    • 당뇨 여부
    본 결과는 AI 예측 결과이며
    실제 의학적 진단을 대체하지 않습니다.
    """
)
# =========================
# 입력
# =========================
st.markdown("## 건강검진 정보 입력")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        "나이",
        min_value=0,
        max_value=120,
        value=30
    )
    sex_text = st.selectbox(
        "성별",
        ["남성", "여성"]
    )
    HE_alt = st.number_input(
        "ALT 수치",
        min_value=0.0,
        value=25.0
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
# =========================
# 변수 변환
# =========================
# 학습 데이터 기준
sex = 1 if sex_text == "남성" else 2
# 데이터 확인 결과:
# 당뇨 있음 = 1
# 당뇨 없음 = 8
DE1_pr = 1 if diabetes_text == "있음" else 8
# =========================
# 예측
# =========================
if st.button("위험 수준 확인하기"):
    input_data = np.array([
        [age, sex, HE_alt, HE_ast, DE1_pr]
    ])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(
        input_scaled,
        verbose=0
    )
    prediction_value = float(
        prediction.reshape(-1)[0]
    )
    risk_score = prediction_value * 100
    st.markdown("---")
    st.markdown("## 예측 결과")
    st.progress(
        min(
            max(
                int(risk_score),
                0
            ),
            100
        )
    )
    st.metric(
        label="AI 위험 점수",
        value=f"{risk_score:.1f}점"
    )
    # =====================
    # 3단계 분류
    # =====================
    if prediction_value < 0.40:
        st.success(
            "저위험군으로 예측됩니다."
        )
        st.write(
            "현재 입력값 기준으로 지방간 위험 가능성이 낮게 예측되었습니다."
        )
    elif prediction_value < 0.75:
        st.warning(
            "주의군으로 예측됩니다."
        )
        st.write(
            "일부 위험 요인이 확인되어 생활습관 관리가 권장됩니다."
        )
    else:
        st.error(
            "고위험군으로 예측됩니다."
        )
        st.write(
            "지방간 위험 가능성이 높게 예측되었습니다."
        )
    # =====================
    # 상세 정보
    # =====================
    with st.expander("상세 정보 보기"):
        st.write(
            f"모델 원시 예측값: {prediction_value:.4f}"
        )
        st.write(
            "AI 위험 점수는 모델 출력값을 기반으로 표시되며 "
            "실제 질환 확률을 의미하지 않습니다."
        )
    st.caption(
        "본 결과는 연구용 AI 모델의 예측 결과이며 "
        "의학적 진단을 대체할 수 없습니다."
    )