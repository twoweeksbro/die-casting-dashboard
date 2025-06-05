import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import pickle
import plotly.graph_objects as go
import shap

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


# # 전체 예측 진행
# @st.cache_data
# def pre_predict():
    





# Session State 초기화
st.session_state.setdefault("current_idx", 1000)
st.session_state.setdefault("is_running", False)
st.session_state.is_running = False  # 시작 기본값






# 버튼 인터페이스
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("시작"):
        st.session_state.is_running = True
with col2:
    if st.button("멈춤"):
        st.session_state.is_running = False
with col3:
    if st.button("초기화"):
        st.session_state["current_idx"] = 100
        st.session_state["is_running"] = False



def render_status_box(title, value):
        if value == 1:
            # color = "#FF4B4B"  # 불량 - 빨강
            # color = "#F28B82"  # 불량 - 빨강
            color = "#E57373"  # 불량 - 빨강
            label = "불량"
        else:
            # color = "#4CAF50"  # 정상 - 초록
            # color = "#A5D6A7"  # 정상 - 초록
            color = "#81C784"  # 정상 - 초록
            label = "정상"

        html_code = f"""
        <div style="
            background-color:{color};
            padding:1rem;
            border-radius:10px;
            color:white;
            font-weight:bold;
            text-align:center;
            font-size:1.2rem;
            ">
            <div style="font-size:0.9rem;">{title}</div>
            <div>{label}</div>
        </div>
        """
        st.markdown(html_code, unsafe_allow_html=True)


# KPI 렌더링
def render_dashboard(current_df):
    
    st.subheader("실시간 KPI")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("총 생산량", len(current_df), border=True)
    col2.metric("정상 개수", (current_df["passorfail"] == 0).sum(), border=True)
    col3.metric("불량 개수", (current_df["passorfail"] == 1).sum(), border=True)
    col4.metric("불량률", f"{(current_df['passorfail'].mean() * 100):.2f}%", border=True)
    col5.metric("연속 정상 개수", 32, border=True)

    st.divider()
    st.subheader('불량 예측')
    

    
    col1, col2, col3 = st.columns(3)
    col1, col2, col3, col4 = st.columns(4)

    y_pred = model.predict(current_df.iloc[[-1]].drop(columns=['id', 'passorfail', 'datetime']))[0]
    y_proba = model.predict_proba(current_df.iloc[[-1]].drop(columns=['id', 'passorfail', 'datetime']))[0][1]

    # col1.metric("예측 결과", y_pred)
    # col2.metric("불량 확률", y_proba)
    # col3.metric("실제 결과", current_df.iloc[-1]['passorfail'])
    
    with col1:
        render_status_box("예측 결과", y_pred)

    with col2:
        st.metric("불량 확률", f"{y_proba:.2f}")

    with col3:
        render_status_box("실제 결과", current_df.iloc[-1]['passorfail'])
    
    with col4:
         # 게이지 차트로 불량 확률 시각화
        # fig = go.Figure(go.Indicator(
        #     mode="gauge+number",
        #     value=y_proba * 100,
        #     title={'text': "예측 불량 확률 (%)"},
        #     gauge={
        #         'axis': {'range': [0, 100]},
        #         'bar': {'color': "red"},
        #         'steps': [
        #             {'range': [0, 30], 'color': "lightgreen"},
        #             {'range': [30, 70], 'color': "yellow"},
        #             {'range': [70, 100], 'color': "red"}
        #         ],
        #     }
        # ))
        # st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=y_proba * 100,
            number={'font': {'color': '#31333F'}},
            title={'text': "예측 불량 확률 (%)", 'font': {'color': '#31333F'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#31333F'},
                'bar': {'color': "#FF4B4B"},
                'bgcolor': "#F0F2F6",  # secondary background
                'borderwidth': 2,
                'bordercolor': "#E0E0E0",
                'steps': [
                    {'range': [0, 30], 'color': "#DFF5E3"},      # 낮음 - 옅은 초록
                    {'range': [30, 70], 'color': "#FFEAA7"},     # 중간 - 노랑
                    {'range': [70, 100], 'color': "#FFCCCC"}     # 높음 - 옅은 빨강
                ],
            }
        ))

        fig.update_layout(
            height=250,
            margin=dict(t=20, b=0, l=0, r=0),
            paper_bgcolor="#FFFFFF",  # 배경색
            font=dict(color="#31333F")  # 텍스트 컬러
        )

        st.plotly_chart(fig, use_container_width=True, key=f"defect_gauge{st.session_state.current_idx}")
            
    
    
    

# 시계열 그래프 렌더링
# def render_time_series(current_df, selected_vars):
#     cols = st.columns(2)
#     for i, var in enumerate(selected_vars):
#         with cols[i % 2]:
#             fig = px.line(current_df.tail(50), x="datetime", y=var, title=var, color='mold_code')
#             st.plotly_chart(fig, use_container_width=True)

# 시계열 그래프 렌더링 (몰드 코드별 탭 추가)
# def render_time_series(current_df, selected_vars):
#     mold_codes = current_df["mold_code"].unique()

#     tabs = st.tabs([f"몰드 {code}" for code in mold_codes])

#     for tab, code in zip(tabs, mold_codes):
#         with tab:
#             st.markdown(f"### 몰드 코드: `{code}`")
#             filtered_df = current_df[current_df["mold_code"] == code]

#             cols = st.columns(2)
#             for i, var in enumerate(selected_vars):
#                 with cols[i % 2]:
#                     fig = px.line(
#                         filtered_df.tail(50),
#                         x="datetime",
#                         y=var,
#                         title=var,
#                         color_discrete_sequence=["#FF4B4B"]
#                     )
#                     unique_key = f"{code}_{var}"
#                     st.plotly_chart(fig, use_container_width=True, key=unique_key)


# def render_time_series(current_df, selected_vars):
#     st.subheader("몰드 코드별 주요 변수 시계열")

#     mold_codes = df["mold_code"].unique()
#     tab_labels = [f"몰드 코드 {code}" for code in mold_codes]
#     tab_objects = st.tabs(tab_labels)  # <- 수정 포인트


    
#     for code, tab in zip(mold_codes, tab_objects):
#         with tab:
#             st.markdown(f"**몰드 코드 {code}**에 대한 최근 시계열 데이터")
#             filtered_df = current_df[current_df["mold_code"] == code]

#             if filtered_df.empty:
#                 st.info("해당 몰드 코드에 대한 데이터가 아직 없습니다.")
#                 continue

#             cols = st.columns(2)
#             for i, var in enumerate(selected_vars):
#                 with cols[i % 2]:
#                     fig = px.line(filtered_df.tail(50), x="datetime", y=var, title=var)
#                     unique_key = f"{code}_{var}_{i}_{st.session_state.current_idx}"
#                     st.plotly_chart(fig, use_container_width=True,key=unique_key)


def render_time_series(current_df, selected_vars):
    st.subheader("몰드 코드별 주요 변수 시계열")

    mold_codes = df["mold_code"].unique()
    tab_labels = ["전체"] + [f"몰드 코드 {code}" for code in mold_codes]
    tab_objects = st.tabs(tab_labels)

    # 전체 탭
    with tab_objects[0]:
        st.markdown("**전체 몰드 코드**의 최근 시계열 데이터")
        cols = st.columns(2)
        for i, var in enumerate(selected_vars):
            with cols[i % 2]:
                fig = px.line(current_df.tail(50), x="datetime", y=var, title=var, color="mold_code")
                unique_key = f"전체_{var}_{i}_{st.session_state.current_idx}"
                st.plotly_chart(fig, use_container_width=True, key=unique_key)

    # 개별 몰드 코드 탭
    for idx, (code, tab) in enumerate(zip(mold_codes, tab_objects[1:])):
        with tab:
            st.markdown(f"**몰드 코드 {code}**에 대한 최근 시계열 데이터")
            filtered_df = current_df[current_df["mold_code"] == code]

            if filtered_df.empty:
                st.info("해당 몰드 코드에 대한 데이터가 아직 없습니다.")
                continue

            cols = st.columns(2)
            for i, var in enumerate(selected_vars):
                with cols[i % 2]:
                    fig = px.line(filtered_df.tail(50), x="datetime", y=var, title=var)
                    unique_key = f"{code}_{var}_{i}_{st.session_state.current_idx}"
                    st.plotly_chart(fig, use_container_width=True, key=unique_key)










# 불량 테이블 렌더링
def render_defect_table(current_df):
    # if not current_df.empty and current_df["passorfail"].iloc[-1] == 1:
    #     st.warning("불량 발생: 최근 데이터에서 불량이 탐지되었습니다.")
    st.subheader("🚨 최근 불량 기록")
    st.dataframe(current_df[current_df["passorfail"] == 1].tail(5), use_container_width=True)

# Placeholder 구역 분리
kpi_placeholder = st.empty()



st.divider()
st.subheader("주요 변수 시계열")

# 변수 선택 (시계열 그래프용)
available_vars = df.select_dtypes("number").columns.tolist()
selected_vars = st.multiselect(
    "시계열로 볼 변수 선택",
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
                # st.error("🚨 불량 발생! 즉시 점검 요망!")
                st.toast("불량 발생: 최근 데이터에서 불량이 탐지되었습니다!")
                # st.balloons()

            with kpi_placeholder.container():
                render_dashboard(current_df)

            with chart_placeholder.container():
                render_time_series(current_df, selected_vars)

            with table_placeholder.container():
                render_defect_table(current_df)

            st.session_state.current_idx += 1
            time.sleep(1)

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
