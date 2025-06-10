import streamlit as st
import pandas as pd
import plotly.express as px
import time
import pickle
import plotly.graph_objects as go
import shap
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config("실시간 주조 공정 시뮬레이션", layout="wide")
st.title("실시간 주조 공정 모니터링 대시보드")

# 데이터 불러오기
@st.cache_data
def load_full_data():
    df = pd.read_csv("data/train_kr.csv")
    df["날짜시간"] = pd.to_datetime(df["시간"] + " " + df["날짜"])
    return df.sort_values("날짜시간").reset_index(drop=True)


# 데이터 불러오기
@st.cache_data
def load_mgmt_data():
    df = pd.read_csv("data/mgmt.csv")
    
    return df


df = load_full_data()
mgmt_df = load_mgmt_data()

# test
korean_to_english = {
    "ID": "id",
    "라인": "line",
    "이름": "name",
    "금형 이름": "mold_name",
    "시간": "time",
    "날짜": "date",
    "생산 수량": "count",
    "작동 여부": "working",
    "비상정지": "emergency_stop",
    "용탕 온도": "molten_temp",
    "설비 운영 주기": "facility_operation_cycleTime",
    "생산 주기": "production_cycletime",
    "저속 구간 속도": "low_section_speed",
    "고속 구간 속도": "high_section_speed",
    "용탕 체적": "molten_volume",
    "주조 압력": "cast_pressure",
    "비스킷 두께": "biscuit_thickness",
    "상부 금형 온도1": "upper_mold_temp1",
    "상부 금형 온도2": "upper_mold_temp2",
    "상부 금형 온도3": "upper_mold_temp3",
    "하부 금형 온도1": "lower_mold_temp1",
    "하부 금형 온도2": "lower_mold_temp2",
    "하부 금형 온도3": "lower_mold_temp3",
    "슬리브 온도": "sleeve_temperature",
    "물리적 강도": "physical_strength",
    "냉각수 온도": "Coolant_temperature",
    "EMS 작동 시간": "EMS_operation_time",
    "등록 시간": "registration_time",
    "불량 여부": "passorfail",
    "시도 신호": "tryshot_signal",
    "몰드 코드": "mold_code",
    "히팅로 작동 여부": "heating_furnace"
}

##
eng_to_kr = {
        "id": "ID",
        "line": "라인",
        "name": "이름",
        "mold_name": "금형 이름",
        "time": "시간",
        "date": "날짜",
        "count": "생산 수량",
        "working": "작동 여부",
        "emergency_stop": "비상정지",
        "molten_temp": "용탕 온도",
        "facility_operation_cycleTime": "설비 운영 주기",
        "production_cycletime": "생산 주기",
        "low_section_speed": "저속 구간 속도",
        "high_section_speed": "고속 구간 속도",
        "molten_volume": "용탕 체적",
        "cast_pressure": "주조 압력",
        "biscuit_thickness": "비스킷 두께",
        "upper_mold_temp1": "상부 금형 온도1",
        "upper_mold_temp2": "상부 금형 온도2",
        "upper_mold_temp3": "상부 금형 온도3",
        "lower_mold_temp1": "하부 금형 온도1",
        "lower_mold_temp2": "하부 금형 온도2",
        "lower_mold_temp3": "하부 금형 온도3",
        "sleeve_temperature": "슬리브 온도",
        "physical_strength": "물리적 강도",
        "Coolant_temperature": "냉각수 온도",
        "EMS_operation_time": "EMS 작동 시간",
        "registration_time": "등록 시간",
        "passorfail": "불량 여부",
        "tryshot_signal": "시도 신호",
        "mold_code": "몰드 코드",
        "heating_furnace": "히팅로 작동 여부"
    }



# 모델 불러오기
@st.cache_data
def load_model():
    with open("model_rf.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


# 모델 불러오기
@st.cache_data
def load_anomaly_model():
    with open("isolation_forest_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


# 모델 불러오기
@st.cache_data
def load_model2():
    # 1. 저장된 모델 불러오기
    with open("xgb_pipeline_model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return pipeline






drop_cols = [
    'ID', '날짜', '시간', '등록 시간',
    '라인', '이름', '금형 이름', '상부 금형 온도3', '하부 금형 온도3'
]

model = load_model()
model_anom = load_anomaly_model()
model2 = load_model2()

# # 전체 예측 진행
# @st.cache_data
# def pre_predict():
    



# shap 
def show_shap_explanation(sample_df, pipeline=model2):
    """
    주어진 단일 샘플과 학습된 파이프라인을 기반으로
    SHAP 기여도 시각화를 Streamlit으로 출력합니다.
    
    Parameters:
    - sample_df: DataFrame, shape (1, n_features) → 단일 샘플
    - pipeline: 학습된 sklearn Pipeline (전처리 + 모델 포함)
    """
    
    sample_df = sample_df.iloc[[-1]].drop(columns=drop_cols).rename(columns=korean_to_english)
    
    # 1. 전처리 및 모델 분리
    X_transformed = pipeline.named_steps['preprocessing'].transform(sample_df)
    model_only = pipeline.named_steps['model']

    # 2. feature 이름 복원
    raw_feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    feature_names = [name.split("__")[-1] for name in raw_feature_names]

    # feature 한글 복원
    feature_names = [eng_to_kr.get(col, col) for col in feature_names]

    
    # 3. 예측 및 확률
    # pred = pipeline.predict(sample_df)[0]
    # proba = pipeline.predict_proba(sample_df)[0][1]

    # 4. SHAP 분석 (TreeExplainer 사용)
    explainer = shap.TreeExplainer(model_only)
    shap_values = explainer.shap_values(X_transformed)

    # 5. Streamlit 출력
    # st.subheader(f"예측 결과: {pred} (불량일 확률: {proba:.2%})")

    # bar plot
    # st.markdown("#### SHAP Bar Plot (상위 기여도)")
    shap_bar = shap.Explanation(values=shap_values[0], data=X_transformed[0], feature_names=feature_names)
    fig_bar, ax = plt.subplots()
    shap.plots.bar(shap_bar, show=False)
    # st.pyplot(fig_bar)
    
    
    
    # 6. SHAP 값을 정리해서 DataFrame 생성
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "value": X_transformed[0],               # 실제 입력 값
        "shap_value": shap_values[0]             # 해당 feature의 기여도
    })

    # 7. 절댓값 기준으로 중요도 순 정렬
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values(by="abs_shap", ascending=False).drop(columns="abs_shap")

    
    # # waterfall plot
    # st.markdown("#### 🌊 SHAP Waterfall Plot")
    # shap_waterfall = shap.Explanation(values=shap_values[0], data=X_transformed[0], feature_names=feature_names)
    # fig_wf, ax = plt.subplots()
    # shap.plots.waterfall(shap_waterfall, show=False)
    # st.pyplot(fig_wf)
    
    return fig_bar, shap_df



# Session State 초기화
st.session_state.setdefault("current_idx", 1200)
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





def render_status_box(title, value, anom=0):
        if value == 1:
            # color = "#FF4B4B"  # 불량 - 빨강
            # color = "#F28B82"  # 불량 - 빨강
            if anom==1:
                color = "#FFD54F"
                label="주의"
            else:
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
            padding:1.5rem;
            margin: 5px;
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
    col2.metric("정상 개수", (current_df["불량 여부"] == 0).sum(), border=True)
    col3.metric("불량 개수", (current_df["불량 여부"] == 1).sum(), border=True)
    col4.metric("불량률", f"{(current_df['불량 여부'].mean() * 100):.2f}%", border=True)
    col5.metric("최근 100개 불량률", f"{(current_df.tail(100)['불량 여부'].mean() * 100):.2f}%", border=True)


    st.divider()
    st.subheader('불량 예측')
    
    # col1, col2, col3 = st.columns([1,1,2])
    col1, col2= st.columns([1,1])

    # RF model
    # y_pred = model.predict(current_df.iloc[[-1]].drop(columns=['id', 'passorfail', 'datetime']))[0]
    # y_proba = model.predict_proba(current_df.iloc[[-1]].drop(columns=['id', 'passorfail', 'datetime']))[0][1]
    
    # XGB
    y_pred = model2.predict(current_df.iloc[[-1]].drop(columns=drop_cols).rename(columns=korean_to_english))[0]
    y_proba = model2.predict_proba(current_df.iloc[[-1]].drop(columns=drop_cols).rename(columns=korean_to_english))[0][1]

    # col1.metric("예측 결과", y_pred)
    # col2.metric("불량 확률", y_proba)
    # col3.metric("실제 결과", current_df.iloc[-1]['passorfail'])
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=y_proba * 100,
            number={
                'suffix': '%',
                'font': {'size': 32, 'color': "#333333"}
            },
            gauge={
                'shape': "bullet",
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "#dddddd"
                },
                'bar': {'color': "#E53935"},  # 빨간색 바
                'bgcolor': "#f0f0f0",         # 전체 배경
                'borderwidth': 0,
                'steps': []                   # 단계 제거
            },
            # title={'text': "<b>불량 확률</b>", 'font': {'size': 18, 'color': "#444444"}}
        ))

        fig.update_layout(
            height=160,
            margin=dict(t=20, b=20, l=20, r=20),
            paper_bgcolor="#ffffff",
            font=dict(color="#333333", size=14)
        )

        st.plotly_chart(fig, use_container_width=True, key=f"defect_linear_gauge_{st.session_state.current_idx}")
        render_status_box("불량 예측 결과", y_pred)
         # render_status_box("실제 결과", current_df.iloc[-1]['passorfail'])
        # 테스트 데이터에서 수치형 추출
        
        # num_test = current_df.iloc[[-1]].select_dtypes(include=['int64', 'float64']).drop(columns=['passorfail'])
        num_test = current_df.iloc[[-1]].select_dtypes(include=['int64', 'float64']).drop(columns=['불량 여부']).rename(columns=korean_to_english)
        # num_test = df.iloc[[-1]].select_dtypes(include=['int64', 'float64']).drop(columns=['passorfail'])

        # 이상치 예측
        pred = model_anom.predict(num_test)
        # pred = loaded_model.predict(num_test)
        pred = np.where(pred == -1, 1, 0)
        render_status_box("이상 탐지 결과", pred, 1)
        render_status_box("실제 값", current_df.iloc[-1]['불량 여부'])

    with col2:
        
        tab1, tab2 = st.tabs(["시각화", "데이터"])
        fig_bar, shap_df = show_shap_explanation(current_df)
        # 탭 1: 시각화
        with tab1:
            st.markdown("#### SHAP Bar Plot (Top 기여도)")
            st.pyplot(fig_bar)

        # 탭 2: 데이터 표시
        with tab2:
            # 8. Streamlit 표로 출력
            st.markdown("#### SHAP 기여도 표 (상위 항목)")
            st.dataframe(shap_df.head(10), use_container_width=True)

        
        
        # st.markdown('#### 예측 결과에 대한 주요 변수 TOP3')
        # st.metric('실제 값', current_df['passorfail'])
    
    # with col3:
    #    show_shap_explanation(current_df)
       


        
            
    

# 관리도 렌더링
def render_mgmt(current_df):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('불량 추이')
        
        # 최근 100개 데이터 기준
        
        total_defect_rate = (current_df['불량 여부']==1).mean()
        rolling_defect_rate = current_df["불량 여부"].rolling(window=100).mean()
        x = current_df['날짜시간']
        
        # fig = go.Figure()

        # # 전체 누적 불량률 (수평선)
        # fig.add_trace(go.Scatter(
        #     x=x,
        #     y=[total_defect_rate] * len(x),
        #     mode="lines",
        #     name="전체 누적 불량률",
        #     line=dict(color="black", dash="dot")
        # ))

        # # 최근 100개 이동 불량률
        # fig.add_trace(go.Scatter(
        #     x=x,
        #     y=rolling_defect_rate,
        #     mode="lines",
        #     name="최근 100개 불량률",
        #     line=dict(color="red")
        # ))

        # fig.update_layout(
        #     title="전체 vs 최근 100개 불량률 비교",
        #     xaxis_title="Index",
        #     yaxis_title="불량률",
        #     yaxis=dict(range=[0, 1]),
        #     template="plotly_white"
        # )

        # st.plotly_chart(fig, use_container_width=True)
        
        # matplotlib
        
        import matplotlib.pyplot as plt
        plt.rcParams['font.family'] ='Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] =False


        fig, ax = plt.subplots(figsize=(12,6))
        
        
        # 전체 누적 불량률 (수평선)
        ax.plot(x, [total_defect_rate] * len(x), linestyle='dotted', color='black', label='전체 누적 불량률')

        # 최근 100개 이동 불량률
        ax.plot(x, rolling_defect_rate, color='red', label='최근 100개 불량률')
        
        ax.set_title("전체 vs 최근 100개 불량률 비교")
        ax.set_xlabel("Index")
        ax.set_ylabel("불량률")
        ax.set_ylim(0, 0.5)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        
                
        
    
    with col2:

        st.subheader("최근 불량 기록")
        current_df.drop([])
        st.dataframe(current_df[current_df["불량 여부"] == 1].tail(10), use_container_width=True)
    


# 몰드 코드 목록 (전체 포함)
mold_codes = df["몰드 코드"].unique().tolist()
mold_codes = ["전체"] + mold_codes

with st.sidebar:
    st.markdown("### 몰드코드 선택")
    selected_code = st.radio("",mold_codes, key="mold_code_selector", index=0)
    st.divider()





def render_time_series(current_df, selected_vars):
    # 가장 최근 몰드 코드
    latest_mold_code = current_df["몰드 코드"].iloc[-1]
    st.markdown(f"### 🔴 현재 데이터의 몰드 코드: `{latest_mold_code}`")

    # 선택된 몰드 코드 필터링
    if selected_code == "전체":
        filtered_df = current_df
        st.markdown("**전체 몰드 코드**의 최근 시계열 데이터")
        color = "몰드 코드"
    else:
        filtered_df = current_df[current_df["몰드 코드"] == selected_code]
        st.markdown(f"**몰드 코드 {selected_code}**에 대한 최근 시계열 데이터")
        color = None

    if filtered_df.empty:
        st.info("해당 몰드 코드에 대한 데이터가 아직 없습니다.")
        return

    cols = st.columns(2)
    for i, var in enumerate(selected_vars):
        with cols[i % 2]:
            df_tail = filtered_df.tail(50)

            # 평균, 표준편차 계산
            mean_val = df_tail[var].mean()
            std_val = df_tail[var].std()
            ucl = mean_val + 3 * std_val
            lcl = mean_val - 3 * std_val

            # 시계열 그래프 생성
            fig = px.line(df_tail, x="날짜시간", y=var, title=var, color=color)

            # UCL, LCL 수평선 추가
            fig.add_trace(go.Scatter(
                x=df_tail["날짜시간"], y=[ucl] * len(df_tail),
                mode="lines", name="UCL (μ+3σ)",
                line=dict(dash="dash", color="red")
            ))
            fig.add_trace(go.Scatter(
                x=df_tail["날짜시간"], y=[lcl] * len(df_tail),
                mode="lines", name="LCL (μ−3σ)",
                line=dict(dash="dash", color="blue")
            ))

            unique_key = f"{selected_code}_{var}_{i}_{st.session_state.current_idx}"
            st.plotly_chart(fig, use_container_width=True, key=unique_key)






# 불량 테이블 렌더링
def render_defect_table(current_df):
    # if not current_df.empty and current_df["passorfail"].iloc[-1] == 1:
    #     st.warning("불량 발생: 최근 데이터에서 불량이 탐지되었습니다.")
    st.subheader("🚨 최근 불량 기록")
    st.dataframe(current_df[current_df["불량 여부"] == 1].tail(5), use_container_width=True)



# Placeholder 구역 분리
kpi_placeholder = st.empty()
st.divider()


# shap 구역
table_placeholder = st.empty()
st.divider()


# 불량 추이 구역
mgnt_placeholder = st.empty()
st.divider()

# 주요 변수 시계열 구역
st.subheader("주요 변수 시계열")

# 변수 선택 (시계열 그래프용)
available_vars = df.select_dtypes("number").columns.tolist()
selected_vars = st.multiselect(
    "시계열로 볼 변수 선택",
    available_vars,
    default=["용탕 온도", "주조 압력", "저속 구간 속도", "상부 금형 온도1"]
)

chart_placeholder = st.empty()


monitor_placeholder = st.empty()






group_dict = {
    "생산 상태 및 장비 조건": [
        '작동 여부', '생산 수량','비상정지',
        '시도 신호', '히팅로 작동 여부','라인'
    ],
    "온도 관련": [
        '상부 금형 온도1', '상부 금형 온도2', '상부 금형 온도3',
        '용탕 온도', '슬리브 온도', '냉각수 온도',
        '하부 금형 온도1', '하부 금형 온도2', '하부 금형 온도3'
        
    ],
    "성형 공정 (속도/압력/두께)": [
        '저속 구간 속도', '고속 구간 속도', '주조 압력',
        '용탕 체적', '비스킷 두께' 
    ],
    "시간 관련": [
        '설비 운영 주기', '생산 주기', 'EMS 작동 시간'
    ]
}




# 그룹별 변수 선택 (멀티셀렉트)
selected_vars_per_group = {}
for group, vars in group_dict.items():
    selected = st.sidebar.multiselect(f" {group}", vars, default=[vars[0]])
    selected_vars_per_group[group] = selected


# 렌더링 함수 잘 되는데 지금 empty
def render_more_data(current_df):
    for group_name, variables in group_dict.items():
        st.markdown(f"## {group_name}")
        left, right = st.columns([5, 5])
        
        # metric 카드 (좌측)
        with left:
            n_cols = 3
            for i in range(0, len(variables), n_cols):
                row_vars = variables[i:i + n_cols]
                cols = st.columns(len(row_vars))
                for col, var in zip(cols, row_vars):
                    value = current_df.iloc[-1][var]
                    display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
                    col.metric(label=var, value=display, border=True)
        
        # 그래프 (우측)
        with right:
            for selected_var in selected_vars_per_group.get(group_name, []):
                st.markdown(f"### {selected_var} 시계열")
                fig = px.line(current_df.tail(50), x="날짜시간", y=selected_var, title=selected_var)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()





# def render_more_data(current_df):
#     container_1 = st.empty()
#     container_2 = st.empty()
#     container_3 = st.empty()
#     container_4 = st.empty()

#     # ① 생산 상태 및 장비 조건
#     with container_1.container():
#         st.markdown("## 생산 상태 및 장비 조건")
#         left, right = st.columns([6, 4])
#         vars = group_dict["생산 상태 및 장비 조건"]

#         with left:
#             for i in range(0, len(vars), 3):
#                 row_vars = vars[i:i + 3]
#                 cols = st.columns(len(row_vars))
#                 for col, var in zip(cols, row_vars):
#                     value = current_df.iloc[-1][var]
#                     display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
#                     col.metric(label=var, value=display, border=True)

#         with right:
#             for selected_var in selected_vars_per_group.get("생산 상태 및 장비 조건", []):
#                 st.markdown(f"### {selected_var} 시계열")
#                 fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
#                 st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_var}_{st.session_state.current_idx}")

#         st.divider()

#     # ② 온도 관련
#     with container_2.container():
#         st.markdown("## 온도 관련")
#         left, right = st.columns([6, 4])
#         vars = group_dict["온도 관련"]

#         with left:
#             for i in range(0, len(vars), 3):
#                 row_vars = vars[i:i + 3]
#                 cols = st.columns(len(row_vars))
#                 for col, var in zip(cols, row_vars):
#                     value = current_df.iloc[-1][var]
#                     display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
#                     col.metric(label=var, value=display, border=True)

#         with right:
#             for selected_var in selected_vars_per_group.get("온도 관련", []):
#                 st.markdown(f"### {selected_var} 시계열")
#                 fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
#                 st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_var}_{st.session_state.current_idx}")

#         st.divider()

#     # ③ 성형 공정 (속도/압력/두께)
#     with container_3.container():
#         st.markdown("## 성형 공정 (속도/압력/두께)")
#         left, right = st.columns([6, 4])
#         vars = group_dict["성형 공정 (속도/압력/두께)"]

#         # 왼쪽: 메트릭 카드
#         with left:
#             for i in range(0, len(vars), 3):
#                 row_vars = vars[i:i + 3]
#                 cols = st.columns(len(row_vars))
#                 for col, var in zip(cols, row_vars):
#                     value = current_df.iloc[-1][var]
#                     display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
#                     col.metric(label=var, value=display, border=True)

#         # 오른쪽: 시계열 그래프
#         with right:
#             for selected_var in selected_vars_per_group.get("성형 공정 (속도/압력/두께)", []):
#                 st.markdown(f"### {selected_var} 시계열")
#                 fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
#                 st.plotly_chart(fig, use_container_width=True)

#         st.divider()

#     # ④ 시간 관련
#     with container_4.container():
#         st.markdown("## 시간 관련")
#         left, right = st.columns([6, 4])
#         vars = group_dict["시간 관련"]

#         with left:
#             for i in range(0, len(vars), 3):
#                 row_vars = vars[i:i + 3]
#                 cols = st.columns(len(row_vars))
#                 for col, var in zip(cols, row_vars):
#                     value = current_df.iloc[-1][var]
#                     display = f"{value:.2f}" if pd.api.types.is_numeric_dtype(current_df[var]) else value
#                     col.metric(label=var, value=display, border=True)

#         with right:
#             for selected_var in selected_vars_per_group.get("시간 관련", []):
#                 st.markdown(f"### {selected_var} 시계열")
#                 fig = px.line(current_df.tail(50), x="datetime", y=selected_var, title=selected_var)
#                 st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_var}_{st.session_state.current_idx}")

#         st.divider()


# 실시간 시뮬레이션
if selected_vars:
    if st.session_state.is_running:
        while st.session_state.current_idx < len(df):
            current_df = df.iloc[:st.session_state.current_idx]

            if not current_df.empty and current_df["불량 여부"].iloc[-1] == 1:
                # st.error("🚨 불량 발생! 즉시 점검 요망!")
                st.toast("불량 발생: 최근 데이터에서 불량이 탐지되었습니다!")
                # st.balloons()

            with kpi_placeholder.container():
                render_dashboard(current_df)

            # with table_placeholder.container():
            #     show_shap_explanation(current_df)
                
            with mgnt_placeholder.container():
                render_mgmt(current_df)
            
            
            with chart_placeholder.container():
                with st.expander("주요 변수 시계열 보기"):
                    render_time_series(current_df, selected_vars)


                
            with monitor_placeholder.container():
                with st.expander("더 많은 데이터 보기"):
                    render_more_data(current_df)
                

            st.session_state.current_idx += 1
            time.sleep(1)

            if not st.session_state.is_running:
                break
    else:
        current_df = df.iloc[:st.session_state.current_idx]
        with kpi_placeholder.container():
            render_dashboard(current_df)
            
        with mgnt_placeholder.container():
            render_mgmt(current_df)
            
        with chart_placeholder.container():
            with st.expander("주요 변수 시계열 보기"):
                render_time_series(current_df, selected_vars)
                
                
        with monitor_placeholder.container():
                with st.expander("더 많은 데이터 보기"):
                    render_more_data(current_df)


