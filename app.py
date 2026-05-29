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
    max-width: 720px;
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
    현재 모델 입력 변수

    • 나이
    • 성별
    • ALT
    • AST
    • 당뇨 여부

    식이 및 운동 항목은 향후 라이프스타일 반영 모델 업데이트를 위한 프로토타입 입력 항목입니다.
    """
)

# =========================
# 기본 건강검진 정보 입력
# =========================

st.markdown("## 건강검진 정보 입력")

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
# 라이프스타일 입력
# =========================

# =========================
# 식습관 입력
# =========================

st.markdown("## 식습관 정보 입력")

st.caption(
    "현재 모델 예측에는 직접 반영되지 않으며, 향후 라이프스타일 반영 모델 업데이트를 위한 입력 항목입니다."
)

high_calorie_meals = st.slider(
    "기름지거나 고열량인 식사를 일주일에 몇 끼 정도 하나요? "
    "(예: 치킨, 피자, 햄버거, 돈까스, 삼겹살, 마라탕, 라면, 튀김류 등)",
    min_value=0,
    max_value=21,
    value=3,
    step=1
)

snack_frequency = st.slider(
    "군것질이나 단 음식을 일주일에 몇 회 정도 먹나요? "
    "(예: 과자, 초콜릿, 케이크, 아이스크림, 빵, 디저트류)",
    min_value=0,
    max_value=21,
    value=3,
    step=1
)

sugary_drink_frequency = st.slider(
    "당이 들어간 음료를 일주일에 몇 회 정도 마시나요? "
    "(예: 탄산음료, 달달한 커피, 버블티, 에너지드링크, 과일주스)",
    min_value=0,
    max_value=21,
    value=2,
    step=1
)

late_night_meals = st.slider(
    "야식 또는 밤 10시 이후 식사를 일주일에 몇 회 정도 하나요?",
    min_value=0,
    max_value=14,
    value=1,
    step=1
)

regular_meals = st.selectbox(
    "식사 시간은 규칙적인 편인가요?",
    [
        "대체로 규칙적임",
        "가끔 불규칙함",
        "자주 불규칙함"
    ]
)

# =========================
# 운동 습관 입력
# =========================

st.markdown("## 운동 습관 정보 입력")

light_exercise_days = st.slider(
    "가벼운 운동을 일주일에 며칠 정도 하나요? "
    "(예: 산책, 스트레칭, 가벼운 자전거)",
    min_value=0,
    max_value=7,
    value=2,
    step=1
)

moderate_exercise_days = st.slider(
    "중등도 운동을 일주일에 며칠 정도 하나요? "
    "(예: 빠르게 걷기, 조깅, 수영, 등산)",
    min_value=0,
    max_value=7,
    value=2,
    step=1
)

vigorous_exercise_days = st.slider(
    "고강도 운동을 일주일에 며칠 정도 하나요? "
    "(예: 달리기, HIIT, 축구, 농구, 웨이트 트레이닝)",
    min_value=0,
    max_value=7,
    value=0,
    step=1
)

exercise_minutes = st.slider(
    "운동하는 날의 평균 운동 시간(분)",
    min_value=0,
    max_value=180,
    value=30,
    step=10
)

sedentary_hours = st.slider(
    "하루 평균 앉아서 보내는 시간(시간)",
    min_value=0,
    max_value=16,
    value=6,
    step=1
)
# =========================
# 변수 변환
# =========================

sex = 1 if sex_text == "남성" else 2

# 학습 데이터 기준:
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

    if prediction_value < 0.60:

        st.success(
            "저위험군으로 예측됩니다."
        )

        st.write(
            "현재 입력값 기준으로 지방간 위험 가능성이 낮게 예측되었습니다."
        )

    elif prediction_value < 0.85:

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

    if diabetes_text == "있음":

        st.info(
            "당뇨 여부는 본 모델의 예측 결과에 크게 반영될 수 있습니다. "
            "따라서 당뇨가 있는 경우 AI 위험 점수가 높게 산출될 수 있으며, "
            "이 결과는 실제 진단이 아닌 참고용 예측값입니다."
        )

    st.markdown("### 입력된 생활습관 정보")

    st.markdown("### 입력된 생활습관 정보")

st.write(f"고열량·기름진 식사: 주 {high_calorie_meals}회")
st.write(f"군것질·디저트 섭취: 주 {snack_frequency}회")
st.write(f"당 함유 음료 섭취: 주 {sugary_drink_frequency}회")
st.write(f"야식: 주 {late_night_meals}회")
st.write(f"식사 규칙성: {regular_meals}")

st.write("---")

st.write(f"가벼운 운동: 주 {light_exercise_days}일")
st.write(f"중등도 운동: 주 {moderate_exercise_days}일")
st.write(f"고강도 운동: 주 {vigorous_exercise_days}일")
st.write(f"운동일 평균 운동 시간: {exercise_minutes}분")
st.write(f"하루 평균 앉아있는 시간: {sedentary_hours}시간")
st.caption(
        "생활습관 정보는 현재 예측값 계산에는 포함되지 않으며, 향후 라이프스타일 반영 모델 업데이트 시 활용될 예정입니다."
    )

with st.expander("상세 정보 보기"):

        st.write(
            f"모델 원시 예측값: {prediction_value:.4f}"
        )

        st.write(
            "AI 위험 점수는 모델 출력값을 기반으로 표시되며 실제 질환 확률을 의미하지 않습니다."
        )

        st.write(
            "학습 데이터의 변수 분포 특성상 당뇨 여부가 예측 결과에 민감하게 작용할 수 있습니다."
        )

st.caption(
        "본 결과는 연구용 AI 모델의 예측 결과이며 의학적 진단을 대체할 수 없습니다."
    )