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
st.session_state.setdefault("current_idx", 100)
st.session_state.setdefault("is_running", False)





with st.sidebar:
    st.markdown("## 시뮬레이션 제어")
    st.markdown("\n")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("START", help="시작"):
            st.session_state.is_running = True
    with col2:
        if st.button("STOP", help="멈춤"):
            st.session_state.is_running = False
    with col3:
        if st.button("RESET", help="초기화"):
            st.session_state["current_idx"] = 100
            st.session_state["is_running"] = False
            
    st.divider()





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
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=y_proba * 100,
            number={'font': {'color': '#FFFFFF'}},
            title={'text': "예측 불량 확률 (%)", 'font': {'color': '#FFFFFF'}},
            delta={'reference': 50, 'increasing': {'color': "#FF7777"}, 'decreasing': {'color': "#77FF77"}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#FFFFFF'},
                'bar': {'color': "#8F669E"},
                'bgcolor': "#405163",  # 배경색: secondaryBackgroundColor
                'borderwidth': 2,
                'bordercolor': "#1F2730",
                'steps': [
                    {'range': [0, 30], 'color': "#1B5E20"},     # 낮음 - 짙은 초록
                    {'range': [30, 70], 'color': "#FBC02D"},    # 보통 - 진한 앰버
                    {'range': [70, 100], 'color': "#C62828"}    # 높음 - 진한 레드
                ],
                # 바늘처럼 threshold 활용
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': y_proba * 100
                }
            }
        ))

        fig.update_layout(
            height=250,
            margin=dict(t=20, b=0, l=0, r=0),
            paper_bgcolor="#1F2730",  # theme backgroundColor
            font=dict(color="#FFFFFF")  # theme textColor
        )

        st.plotly_chart(fig, use_container_width=True, key=f"defect_gauge{st.session_state.current_idx}")

    with col2:
        render_status_box("예측 결과", y_pred)

    with col3:
        render_status_box("실제 결과", current_df.iloc[-1]['passorfail'])
    
    with col4:
        st.write('hello')
        
            
    
    
    

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

# 잘됨 탭 강조는 안되지만 but 탭이 초기화
# def render_time_series(current_df, selected_vars):
#     st.subheader("몰드 코드별 주요 변수 시계열")

#     mold_codes = df["mold_code"].unique()
    
#     # tab_labels = ["전체"] + [f"몰드 코드 {code}" for code in mold_codes]
    
#     # 탭 라벨: 현재 몰드 코드만 🔴 강조
#     latest_mold_code = current_df["mold_code"].iloc[-1]  # 현재 시점에서 가장 최근 몰드 코드

#     tab_labels = [f"전체"] + [
#         f"🔴 몰드 코드 {code}" if code == latest_mold_code else f"몰드 코드 {code}"
#         for code in mold_codes
#     ]
    
    
#     tab_objects = st.tabs(tab_labels)

#     # 전체 탭
#     with tab_objects[0]:
#         st.markdown("**전체 몰드 코드**의 최근 시계열 데이터")
#         cols = st.columns(2)
#         for i, var in enumerate(selected_vars):
#             with cols[i % 2]:
#                 fig = px.line(current_df.tail(50), x="datetime", y=var, title=var, color="mold_code")
#                 unique_key = f"전체_{var}_{i}_{st.session_state.current_idx}"
#                 st.plotly_chart(fig, use_container_width=True, key=unique_key)

#     # 개별 몰드 코드 탭
#     for idx, (code, tab) in enumerate(zip(mold_codes, tab_objects[1:])):
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
#                     st.plotly_chart(fig, use_container_width=True, key=unique_key)


# 몰드 코드 목록 (전체 포함)
mold_codes = df["mold_code"].unique().tolist()
mold_codes = ["전체"] + mold_codes

with st.sidebar:
    st.markdown("### 몰드코드 선택")
    selected_code = st.radio("",mold_codes, key="mold_code_selector", index=0)
    st.divider()





def render_time_series(current_df, selected_vars):
    st.subheader("몰드 코드별 주요 변수 시계열")


    # 가장 최근 몰드 코드
    latest_mold_code = current_df["mold_code"].iloc[-1]
    st.markdown(f"### 🔴 현재 데이터의 몰드 코드: `{latest_mold_code}`")

    # 사이드바 라디오 버튼으로 선택

    if selected_code == "전체":
        filtered_df = current_df
        st.markdown("**전체 몰드 코드**의 최근 시계열 데이터")
        color = "mold_code"
    else:
        filtered_df = current_df[current_df["mold_code"] == selected_code]
        st.markdown(f"**몰드 코드 {selected_code}**에 대한 최근 시계열 데이터")
        color = None  # 단일 색상

    if filtered_df.empty:
        st.info("해당 몰드 코드에 대한 데이터가 아직 없습니다.")
        return

    cols = st.columns(2)
    for i, var in enumerate(selected_vars):
        with cols[i % 2]:
            fig = px.line(filtered_df.tail(50), x="datetime", y=var, title=var, color=color)
            unique_key = f"{selected_code}_{var}_{i}_{st.session_state.current_idx}"
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

monitor_placeholder = st.empty()






group_dict = {
    "생산 상태 및 장비 조건": [
        'working', 'count','emergency_stop',
        'tryshot_signal', 'heating_furnace','line'
    ],
    "온도 관련": [
        'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
        'molten_temp', 'sleeve_temperature', 'Coolant_temperature',
        'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3'
        
    ],
    "성형 공정 (속도/압력/두께)": [
        'low_section_speed', 'high_section_speed', 'cast_pressure',
        'molten_volume', 'biscuit_thickness' 
    ],
    "시간 관련": [
        'facility_operation_cycleTime', 'production_cycletime', 'EMS_operation_time'
    ]
}



# # 더 많은 변수 시각화 렌더링
# def render_more_data(current_df):
#     # 그룹별 변수 정의

#     # 각 그룹에 대해 보여주기
#     for group_name, variables in group_dict.items():
#         st.markdown(f"##  {group_name}")
#         cols = st.columns(len(variables))

#         for i, var in enumerate(variables):
#             value = current_df.iloc[-1][var]
#             cols[i].metric(label=var, value=f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value, border=)

#         if selected_var in variables:
#             st.markdown(f"### ⏳ {selected_var} 시계열")
#             fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
#             st.plotly_chart(fig, use_container_width=True)

#         st.divider()
    

# 줄 바꿈 까지는 굿.
# def render_more_data(current_df):
#     for group_name, variables in group_dict.items():
#         st.markdown(f"##  {group_name}")

#         # 한 줄에 3개씩 출력
#         n_cols = 3
#         for i in range(0, len(variables), n_cols):
#             row_vars = variables[i:i + n_cols]
#             cols = st.columns(len(row_vars))
#             for col, var in zip(cols, row_vars):
#                 value = current_df.iloc[-1][var]
#                 display_value = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
#                 col.metric(label=var, value=display_value, border=True)

#         # 선택한 변수에 대해 시계열 그래프 출력
#         if selected_var in variables:
#             st.markdown(f"### ⏳ {selected_var} 시계열")
#             fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
#             st.plotly_chart(fig, use_container_width=True)

#         st.divider()

# 현재 선택된 변수
# all_vars = sum(group_dict.values(), [])
# selected_var = st.sidebar.radio("📈 시계열로 볼 변수 선택", all_vars, index=0)


# def render_more_data(current_df):
#     for group_name, variables in group_dict.items():
#         st.markdown(f"## {group_name}")
        
#         # 비율 6:4로 칼럼 분리
#         left, right = st.columns([6, 4])
        
#         # 왼쪽: metric 카드 (한 줄에 3개씩)
#         with left:
#             n_cols = 3
#             for i in range(0, len(variables), n_cols):
#                 row_vars = variables[i:i + n_cols]
#                 cols = st.columns(len(row_vars))
#                 for col, var in zip(cols, row_vars):
#                     value = current_df.iloc[-1][var]
#                     display_value = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
#                     col.metric(label=var, value=display_value, border=True)

#         # 오른쪽: 시계열 그래프
#         with right:
#             if selected_var in variables:
#                 st.markdown(f"### ⏳ {selected_var} 시계열")
#                 fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
#                 st.plotly_chart(fig, use_container_width=True)

#         st.divider()




# group_selected_vars = {}
# for group_name, vars in group_dict.items():
#     group_selected_vars[group_name] = st.sidebar.radio(
#         f" {group_name} 시계열 변수 선택",
#         vars,
#         index=0,
#         key=f"{group_name}_selector"
#     )
    
    
    

# 본문 시각화 함수
# def render_more_data(current_df):
#     for group_name, variables in group_dict.items():
#         st.markdown(f"## {group_name}")
#         left, right = st.columns([5, 5])

#         with left:
#             for i in range(0, len(variables), 3):
#                 row_vars = variables[i:i + 3]
#                 cols = st.columns(len(row_vars))
#                 for col, var in zip(cols, row_vars):
#                     value = current_df.iloc[-1][var]
#                     display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
#                     col.metric(label=var, value=display, border=True)

#         with right:
#             selected_var = group_selected_vars[group_name]
#             st.markdown(f"###  {selected_var} 시계열")
#             fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
#             st.plotly_chart(fig, use_container_width=True)

#         st.divider()


# 그룹별 변수 선택 (멀티셀렉트)
selected_vars_per_group = {}
for group, vars in group_dict.items():
    selected = st.sidebar.multiselect(f" {group}", vars, default=[vars[0]])
    selected_vars_per_group[group] = selected


# 렌더링 함수 잘 되는데 지금 empty
# def render_more_data(current_df):
#     for group_name, variables in group_dict.items():
#         st.markdown(f"## 🔧 {group_name}")
#         left, right = st.columns([5, 5])
        
#         # metric 카드 (좌측)
#         with left:
#             n_cols = 3
#             for i in range(0, len(variables), n_cols):
#                 row_vars = variables[i:i + n_cols]
#                 cols = st.columns(len(row_vars))
#                 for col, var in zip(cols, row_vars):
#                     value = current_df.iloc[-1][var]
#                     display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
#                     col.metric(label=var, value=display, border=True)
        
#         # 그래프 (우측)
#         with right:
#             for selected_var in selected_vars_per_group.get(group_name, []):
#                 st.markdown(f"### ⏳ {selected_var} 시계열")
#                 fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
#                 st.plotly_chart(fig, use_container_width=True)

#         st.divider()



# def render_more_data(current_df):
#     container_1 = st.empty()
#     container_2 = st.empty()
#     container_3 = st.empty()
#     container_4 = st.empty()

#     group_containers = {
#         "생산 상태 및 장비 조건": container_1,
#         "온도 관련": container_2,
#         "성형 공정 (속도/압력/두께)": container_3,
#         "시간 관련": container_4
#     }

#     for group_name, container in group_containers.items():
#         with container.container():
#             st.markdown(f"## 🔧 {group_name}")
#             left, right = st.columns([6, 4])

#             vars = group_dict[group_name]

#             # 좌측: 메트릭 또는 게이지
#             with left:
#                 n_cols = 3
#                 for i in range(0, len(vars), n_cols):
#                     row_vars = vars[i:i + n_cols]
#                     cols = st.columns(len(row_vars))

#                     for col, var in zip(cols, row_vars):
#                         value = current_df.iloc[-1][var]
#                         if group_name == "성형 공정 (속도/압력/두께)" and pd.api.types.is_numeric_dtype(current_df[var]):
#                             # 게이지 차트
#                             fig = go.Figure(go.Indicator(
#                                 mode="gauge+number",
#                                 value=value,
#                                 title={'text': var},
#                                 gauge={
#                                     'axis': {'range': [0, max(1, value * 1.5)]},
#                                     'bar': {'color': "#FF4B4B"},
#                                     'steps': [
#                                         {'range': [0, value * 0.7], 'color': "#DFF5E3"},
#                                         {'range': [value * 0.7, value * 1.2], 'color': "#FFEAA7"},
#                                         {'range': [value * 1.2, value * 1.5], 'color': "#FFCCCC"}
#                                     ]
#                                 },
#                                 number={'font': {'size': 14}}
#                             ))
#                             col.plotly_chart(fig, use_container_width=True)
#                         else:
#                             # 일반 메트릭
#                             display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
#                             col.metric(label=var, value=display, border=True)

#             # 우측: 시계열 그래프
#             with right:
#                 for selected_var in selected_vars_per_group.get(group_name, []):
#                     st.markdown(f"### ⏳ {selected_var} 시계열")
#                     fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
#                     st.plotly_chart(fig, use_container_width=True)

#             st.divider()

def render_more_data(current_df):
    container_1 = st.empty()
    container_2 = st.empty()
    container_3 = st.empty()
    container_4 = st.empty()

    # ① 생산 상태 및 장비 조건
    with container_1.container():
        st.markdown("## 생산 상태 및 장비 조건")
        left, right = st.columns([6, 4])
        vars = group_dict["생산 상태 및 장비 조건"]

        with left:
            for i in range(0, len(vars), 3):
                row_vars = vars[i:i + 3]
                cols = st.columns(len(row_vars))
                for col, var in zip(cols, row_vars):
                    value = current_df.iloc[-1][var]
                    display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
                    col.metric(label=var, value=display, border=True)

        with right:
            for selected_var in selected_vars_per_group.get("생산 상태 및 장비 조건", []):
                st.markdown(f"### {selected_var} 시계열")
                fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_var}_{st.session_state.current_idx}")

        st.divider()

    # ② 온도 관련
    with container_2.container():
        st.markdown("## 온도 관련")
        left, right = st.columns([6, 4])
        vars = group_dict["온도 관련"]

        with left:
            for i in range(0, len(vars), 3):
                row_vars = vars[i:i + 3]
                cols = st.columns(len(row_vars))
                for col, var in zip(cols, row_vars):
                    value = current_df.iloc[-1][var]
                    display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
                    col.metric(label=var, value=display, border=True)

        with right:
            for selected_var in selected_vars_per_group.get("온도 관련", []):
                st.markdown(f"### {selected_var} 시계열")
                fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_var}_{st.session_state.current_idx}")

        st.divider()

    # ③ 성형 공정 (속도/압력/두께)
    with container_3.container():
        st.markdown("## 성형 공정 (속도/압력/두께)")
        left, right = st.columns([6, 4])
        vars = group_dict["성형 공정 (속도/압력/두께)"]

        # 왼쪽: 메트릭 카드
        with left:
            for i in range(0, len(vars), 3):
                row_vars = vars[i:i + 3]
                cols = st.columns(len(row_vars))
                for col, var in zip(cols, row_vars):
                    value = current_df.iloc[-1][var]
                    display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
                    col.metric(label=var, value=display, border=True)

        # 오른쪽: 시계열 그래프
        with right:
            for selected_var in selected_vars_per_group.get("성형 공정 (속도/압력/두께)", []):
                st.markdown(f"### {selected_var} 시계열")
                fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

    # ④ 시간 관련
    with container_4.container():
        st.markdown("## 시간 관련")
        left, right = st.columns([6, 4])
        vars = group_dict["시간 관련"]

        with left:
            for i in range(0, len(vars), 3):
                row_vars = vars[i:i + 3]
                cols = st.columns(len(row_vars))
                for col, var in zip(cols, row_vars):
                    value = current_df.iloc[-1][var]
                    display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
                    col.metric(label=var, value=display, border=True)

        with right:
            for selected_var in selected_vars_per_group.get("시간 관련", []):
                st.markdown(f"### {selected_var} 시계열")
                fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_var}_{st.session_state.current_idx}")

        st.divider()


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
                
            with monitor_placeholder.container():
                render_more_data(current_df)
                

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
