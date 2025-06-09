import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("불량 확률 예측 시뮬레이터")

@st.cache_data
def load_model():
    with open("model_rf.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

model = load_model()
df = load_data()

group_dict = {
    "생산 상태 및 장비 조건": ['working', 'count', 'emergency_stop', 'tryshot_signal', 'heating_furnace', 'line'],
    "온도 관련": ['upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
                 'molten_temp', 'sleeve_temperature', 'Coolant_temperature',
                 'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3'],
    "성형 공정 (속도/압력/두께)": ['low_section_speed', 'high_section_speed', 'cast_pressure',
                                'molten_volume', 'biscuit_thickness', 'physical_strength'],
    "시간 관련": ['facility_operation_cycleTime', 'production_cycletime', 'EMS_operation_time',
               'time', 'date', 'registration_time']
}

X = df.drop(columns=["id", "passorfail"])
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
all_features = num_cols + cat_cols

# ✅ 모델 사용 변수 전체 선택 + 사용자 제거 가능
st.sidebar.header("입력 변수 선택")
selected_vars = st.sidebar.multiselect("모델 입력 변수 선택", all_features, default=all_features)

input_data = {}
st.header("입력값 설정")
cols = st.columns(4)

for i, (group, vars) in enumerate(group_dict.items()):
    with cols[i]:
        st.subheader(group)
        for var in vars:
            if var not in selected_vars:
                continue
            if var in num_cols:
                q10, q90 = df[var].quantile(0.1), df[var].quantile(0.9)
                default = df[var].mean()
                input_data[var] = st.slider(f"{var}", float(q10), float(q90), float(default))
            elif var in cat_cols:
                options = sorted(df[var].dropna().unique())
                input_data[var] = st.selectbox(f"{var}", options)

if st.button("불량 확률 예측하기"):
    input_df = pd.DataFrame([input_data])

    for col in all_features:
        if col not in input_df.columns:
            input_df[col] = df[col].mean() if col in num_cols else df[col].mode()[0]

    input_df = input_df[all_features]
    pred_prob = model.predict_proba(input_df)[0][1]

    st.subheader("예측된 불량 확률")
    st.metric(label="불량 확률", value=f"{pred_prob*100:.2f}%")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.empty()
    with col2:
        if pred_prob >= 0.7:
            st.markdown("""
            <div style='background-color:#ffcccc; padding:1rem; border-radius:10px;'>
                <strong>불량 확률이 매우 높습니다!</strong><br>주요 공정을 점검하세요.
            </div>
            """, unsafe_allow_html=True)
        elif pred_prob >= 0.4:
            st.markdown("""
            <div style='background-color:#fff3cd; padding:1rem; border-radius:10px;'>
                <strong>불량 위험이 다소 있습니다.</strong><br>주의가 필요합니다.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color:#d4edda; padding:1rem; border-radius:10px;'>
                <strong>불량 확률이 낮습니다.</strong><br>공정이 안정적입니다.
            </div>
            """, unsafe_allow_html=True)
