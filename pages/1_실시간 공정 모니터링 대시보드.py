import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import pickle

st.set_page_config("실시간 주조 공정 시뮬레이션", layout="wide")
st.title("실시간 주조 공정 모니터링 대시보드")
# st.title("실시간 주조 시뮬레이션")

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
    # 모델 불러오기
    with open("model_rf.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model

model = load_model()


# Session State 초기화
st.session_state.setdefault("current_idx", 100)
#     st.session_state.current_idx = 100

st.session_state.setdefault("is_running", False)
# 일단 True로 해놓음
st.session_state.is_running= True

# if "current_idx" not in st.session_state:
#     st.session_state.current_idx = 100
# if "is_running" not in st.session_state:
#     st.session_state.is_running = False






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

# 변수 선택
available_vars = df.select_dtypes("number").columns.tolist()
selected_vars = st.multiselect(
    "시계열로 볼 변수 선택 (최대 4개)",
    available_vars,
    default=["molten_temp", "cast_pressure", "low_section_speed", "upper_mold_temp1"]
)

# 공통 렌더링 함수
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
    
    col1.metric("예측 결과",y_pred)
    col2.metric("불량 확률",y_proba)
    col3.metric("실제 결과",current_df.iloc[-1]['passorfail'])
    
    
    
    st.divider()
    st.subheader("주요 변수 시계열")
    cols = st.columns(2)
    for i, var in enumerate(selected_vars):
        with cols[i % 2]:
            fig = px.line(current_df.tail(100), x="datetime", y=var, title=var)
            st.plotly_chart(fig, use_container_width=True)

    if not current_df.empty and current_df["passorfail"].iloc[-1] == 1:
        st.warning("불량 발생: 최근 데이터에서 불량이 탐지되었습니다.")
    st.subheader("🚨 최근 불량 기록")
    st.dataframe(current_df[current_df["passorfail"] == 1].tail(5), use_container_width=True)




# 실시간 시뮬레이션
placeholder = st.empty()

if selected_vars:
    if st.session_state.is_running:
        while st.session_state.current_idx < len(df):
            current_df = df.iloc[:st.session_state.current_idx]
    
            if not current_df.empty and current_df["passorfail"].iloc[-1] == 1:
                st.toast("불량 발생: 최근 데이터에서 불량이 탐지되었습니다!")
                # st.warning("불량 발생: 최근 데이터에서 불량이 탐지되었습니다.")
            #     # 또는 st.warning("불량 발생: 최근 데이터에서 불량이 탐지되었습니다.")
            
            with placeholder.container():
                render_dashboard(current_df)

            st.session_state.current_idx += 1
            time.sleep(2)

            if not st.session_state.is_running:
                break
    else:
        current_df = df.iloc[:st.session_state.current_idx]
        with placeholder.container():
            render_dashboard(current_df)






# # 현재 시점의 마지막 row (최근 데이터)
# latest_row = df.iloc[st.session_state.current_idx - 1:st.session_state.current_idx].copy()

# # 예측에 사용할 컬럼 선택
# model_features = loaded_model.named_steps['columntransformer'].get_feature_names_out()
# # 전처리 + 예측
# X_latest = latest_row.drop(columns=["id", "passorfail", "datetime", "date", "time"], errors="ignore")

# # 예측 실행
# y_pred = loaded_model.predict(X_latest)[0]
# y_proba = loaded_model.predict_proba(X_latest)[0][1]

# # 결과 출력
# result_text = "정상" if y_pred == 0 else "불량"
# result_color = "green" if y_pred == 0 else "red"

# col1, col2 = st.columns(2)
# col1.metric("예측 결과", result_text)
# col2.metric("불량 확률", f"{y_proba:.2%}")


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import time
# from datetime import datetime

# st.set_page_config("실시간 주조 공정 시뮬레이션", layout="wide")
# st.title("실시간 주조 공정 시뮬레이션")

# # 데이터 불러오기
# @st.cache_data
# def load_full_data():
#     df = pd.read_csv("data/train.csv")
#     df["datetime"] = pd.to_datetime(df["time"] + " " + df["date"])
#     return df.sort_values("datetime").reset_index(drop=True)

# df = load_full_data()

# # Session State 초기화
# if "current_idx" not in st.session_state:
#     st.session_state.current_idx = 100
# if "is_running" not in st.session_state:
#     st.session_state.is_running = False

# # 버튼 인터페이스
# col1, col2, col3 = st.columns(3)
# with col1:
#     if st.button("▶️ 시작"):
#         st.session_state.is_running = True
# with col2:
#     if st.button("⏹️ 멈춤"):
#         st.session_state.is_running = False
# with col3:
#     if st.button("🔄 초기화"):
#         st.session_state.current_idx = 100
#         st.session_state.is_running = False

# # 변수 선택
# available_vars = df.select_dtypes("number").columns.tolist()
# selected_vars = st.multiselect(
#     "📊 시계열로 볼 변수 선택 (최대 4개)",
#     available_vars,
#     default=["molten_temp", "cast_pressure", "low_section_speed", "upper_mold_temp1"]
# )

# # 시뮬레이션 영역
# placeholder = st.empty()

# # 실시간 루프
# if st.session_state.is_running and selected_vars:
#     while st.session_state.current_idx < len(df):
#         current_df = df.iloc[:st.session_state.current_idx]

#         with placeholder.container():
#             st.subheader("📊 실시간 KPI")
#             col1, col2, col3, col4 = st.columns(4)
#             col1.metric("총 생산량", len(current_df))
#             col2.metric("정상 개수", (current_df["passorfail"] == 0).sum())
#             col3.metric("불량 개수", (current_df["passorfail"] == 1).sum())
#             col4.metric("불량률", f"{(current_df['passorfail'].mean() * 100):.2f}%")

#             st.divider()
#             st.subheader("📈 주요 변수 시계열")
#             cols = st.columns(2)
#             for i, var in enumerate(selected_vars):
#                 with cols[i % 2]:
#                     fig = px.line(current_df.tail(100), x="datetime", y=var, title=var)
#                     st.plotly_chart(fig, use_container_width=True)

#             st.subheader("🚨 최근 불량 기록")
#             st.dataframe(current_df[current_df["passorfail"] == 1].tail(5), use_container_width=True)

#         # 업데이트
#         st.session_state.current_idx += 1
#         time.sleep(1)  # ← 여기가 초 단위 (1초마다 새 데이터 표시)

#         # 중간에 사용자가 멈추기를 눌렀다면 break
#         if not st.session_state.is_running:
#             break






# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import time
# from datetime import datetime

# st.set_page_config("실시간 주조 공정 시뮬레이션", layout="wide")
# st.title("실시간 주조 공정 시뮬레이션")

# # 전체 데이터 로드
# @st.cache_data
# def load_full_data():
#     df = pd.read_csv("data/train.csv")
#     df["datetime"] = pd.to_datetime(df["time"] + " " + df["date"])
#     return df.sort_values("datetime").reset_index(drop=True)

# full_df = load_full_data()

# # 시작 index (루프 반복 횟수 조절용)
# start_idx = 100
# max_idx = len(full_df)
# step = 1  # 한 번에 들어오는 row 수
# delay = 5000  # 초 단위 delay

# # 시계열 변수 선택
# available_vars = full_df.select_dtypes("number").columns.tolist()
# selected_vars = st.multiselect(
#     "📊 시계열로 볼 변수 선택 (최대 4개)",
#     available_vars,
#     default=["molten_temp", "cast_pressure", "low_section_speed", "upper_mold_temp1"]
# )

# # 표시 영역 미리 선언
# placeholder = st.empty()

# if selected_vars:
#     for i in range(start_idx, max_idx, step):
#         current_df = full_df.iloc[:i]

#         with placeholder.container():
#             st.subheader("📊 실시간 KPI")
#             col1, col2, col3, col4 = st.columns(4)
#             col1.metric("총 생산량", len(current_df))
#             col2.metric("정상 개수", (current_df["passorfail"] == 0).sum())
#             col3.metric("불량 개수", (current_df["passorfail"] == 1).sum())
#             col4.metric("불량률", f"{(current_df['passorfail'].mean() * 100):.2f}%")

#             st.divider()
#             st.subheader("📈 주요 변수 시계열")
#             cols = st.columns(2)
#             for j, var in enumerate(selected_vars):
#                 with cols[j % 2]:
#                     fig = px.line(current_df.tail(100), x="datetime", y=var, title=var)
#                     st.plotly_chart(fig, use_container_width=True)

#             st.subheader("🚨 최근 불량 기록")
#             st.dataframe(current_df[current_df["passorfail"] == 1].tail(5), use_container_width=True)

#         time.sleep(delay)  # 다음 루프로 넘어가기 전 대기


# # import streamlit as st
# # import pandas as pd
# # import plotly.express as px
# # from datetime import datetime
# # from streamlit_autorefresh import st_autorefresh

# # # ⏱️ 자동 새로고침 (예: 2초마다)
# # st_autorefresh(interval=2000, key="refresh")

# # st.title("🔄 실시간 주조 공정 시뮬레이션")

# # # CSV 로드 (전체 데이터)
# # @st.cache_data
# # def load_full_data():
# #     df = pd.read_csv("data/train.csv")
# #     df["datetime"] = pd.to_datetime(df["time"] + " " + df["date"])
# #     df = df.sort_values("datetime").reset_index(drop=True)
# #     return df

# # full_df = load_full_data()

# # # session_state 로 현재 index 추적
# # if "current_idx" not in st.session_state:
# #     st.session_state.current_idx = 100  # 초기 50개

# # # 한 줄씩 추가 (혹은 여러 줄씩 증가도 가능)
# # increment = 1
# # st.session_state.current_idx += increment

# # # 현재 시점 데이터만 보여주기
# # current_df = full_df.iloc[:st.session_state.current_idx]

# # # KPI 보여주기
# # col1, col2, col3, col4 = st.columns(4)
# # col1.metric("총 생산량", len(current_df))
# # col2.metric("정상 개수", (current_df["passorfail"] == 0).sum())
# # col3.metric("불량 개수", (current_df["passorfail"] == 1).sum())
# # col4.metric("불량률", f"{(current_df['passorfail'].mean() * 100):.2f}%")

# # st.divider()

# # # 📈 시계열 변수 선택
# # selected_vars = st.multiselect(
# #     "📊 시계열로 볼 변수 선택 (최대 4개)", 
# #     current_df.select_dtypes("number").columns.tolist(), 
# #     default=["molten_temp", "cast_pressure", "low_section_speed", "upper_mold_temp1"]
# # )

# # if selected_vars:
# #     st.subheader("📈 시계열 그래프")
# #     cols = st.columns(2)
# #     for i, var in enumerate(selected_vars):
# #         with cols[i % 2]:
# #             fig = px.line(current_df.tail(100), x="datetime", y=var, title=var)
# #             st.plotly_chart(fig, use_container_width=True)

# # # 📝 최근 불량 데이터
# # st.subheader("🚨 최근 불량 기록")
# # st.dataframe(current_df[current_df["passorfail"] == 1].tail(5), use_container_width=True)
