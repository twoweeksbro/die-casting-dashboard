import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import pickle

st.set_page_config("실시간 주조 공정 시뮬레이션", layout="wide")
st.title("실시간 주조 공정 모니터링 대시보드")

# 데이터 불러오기
@st.cache_data
def load_full_data():
    df = pd.read_csv("data/train.csv")
    df["datetime"] = pd.to_datetime(df["time"] + " " + df["date"])
    return df.sort_values("datetime").reset_index(drop=True)

df = load_full_data()

# 모델 불러오기
@st.cache_data
def load_model():
    with open("model_rf.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model

model = load_model()

# Session State 초기화
st.session_state.setdefault("current_idx", 100)
st.session_state.setdefault("is_running", False)
st.session_state.is_running = True  # 시작 기본값

# 버튼 인터페이스
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ 시작"):
        st.session_state.is_running = True
with col2:
    if st.button("⏹️ 멈춤"):
        st.session_state.is_running = False
with col3:
    if st.button("🔄 초기화"):
        st.session_state["current_idx"] = 100
        st.session_state["is_running"] = False



# KPI 렌더링
def render_dashboard(current_df):
    st.subheader("실시간 KPI")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 생산량", len(current_df))
    col2.metric("정상 개수", (current_df["passorfail"] == 0).sum())
    col3.metric("불량 개수", (current_df["passorfail"] == 1).sum())
    col4.metric("불량률", f"{(current_df['passorfail'].mean() * 100):.2f}%")

    st.divider()
    st.subheader('불량 예측')
    col1, col2, col3 = st.columns(3)

    y_pred = model.predict(current_df.iloc[[-1]].drop(columns=['id', 'passorfail', 'datetime']))[0]
    y_proba = model.predict_proba(current_df.iloc[[-1]].drop(columns=['id', 'passorfail', 'datetime']))[0][1]

    col1.metric("예측 결과", y_pred)
    col2.metric("불량 확률", y_proba)
    col3.metric("실제 결과", current_df.iloc[-1]['passorfail'])

# 시계열 그래프 렌더링
def render_time_series(current_df, selected_vars):
    cols = st.columns(2)
    for i, var in enumerate(selected_vars):
        with cols[i % 2]:
            fig = px.line(current_df.tail(100), x="datetime", y=var, title=var)
            st.plotly_chart(fig, use_container_width=True)

# 불량 테이블 렌더링
def render_defect_table(current_df):
    if not current_df.empty and current_df["passorfail"].iloc[-1] == 1:
        st.warning("불량 발생: 최근 데이터에서 불량이 탐지되었습니다.")
    st.subheader("🚨 최근 불량 기록")
    st.dataframe(current_df[current_df["passorfail"] == 1].tail(5), use_container_width=True)

# Placeholder 구역 분리
kpi_placeholder = st.empty()

st.divider()
st.subheader("주요 변수 시계열")

# 변수 선택 (시계열 그래프용)
available_vars = df.select_dtypes("number").columns.tolist()
selected_vars = st.multiselect(
    "시계열로 볼 변수 선택 (최대 4개)",
    available_vars,
    default=["molten_temp", "cast_pressure", "low_section_speed", "upper_mold_temp1"]
)

chart_placeholder = st.empty()
table_placeholder = st.empty()

# 실시간 시뮬레이션
if selected_vars:
    if st.session_state.is_running:
        while st.session_state.current_idx < len(df):
            current_df = df.iloc[:st.session_state.current_idx]

            if not current_df.empty and current_df["passorfail"].iloc[-1] == 1:
                st.toast("불량 발생: 최근 데이터에서 불량이 탐지되었습니다!")
                st.balloons()

            with kpi_placeholder.container():
                render_dashboard(current_df)

            with chart_placeholder.container():
                render_time_series(current_df, selected_vars)

            with table_placeholder.container():
                render_defect_table(current_df)

            st.session_state.current_idx += 1
            time.sleep(2)

            if not st.session_state.is_running:
                break
    else:
        current_df = df.iloc[:st.session_state.current_idx]
        with kpi_placeholder.container():
            render_dashboard(current_df)
        with chart_placeholder.container():
            render_time_series(current_df, selected_vars)
        with table_placeholder.container():
            render_defect_table(current_df)
