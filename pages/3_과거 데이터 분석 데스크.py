# # pages/2_과거_분석_탐산.py
# import streamlit as st
# import pandas as pd
# import plotly.express as px

# st.title("과거 데이터 분석 데스크")

# # 데이터 로드
# @st.cache_data

# def load_data():
#     df = pd.read_csv("data/train.csv")
#     df["datetime"] = pd.to_datetime(df['time'] + " "+ df['date'])

#     df["hour"] = df["datetime"].dt.hour
#     df["weekday"] = df["datetime"].dt.day_name()
#     return df

# df = load_data()

# # 요일별 불량률
# st.subheader("1. 요일별 불량")
# fail_rate = df.groupby("weekday")["passorfail"].mean().reset_index()
# weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# fig1 = px.bar(fail_rate, x="weekday", y="passorfail", title="요일별 불량", category_orders={"weekday": weekday_order})
# st.plotly_chart(fig1, use_container_width=True)

# # 주간 vs 야간 불량
# st.subheader("2. 주간 vs 야간 불량")
# df["shift"] = df["hour"].apply(lambda x: "주간" if 6 <= x < 18 else "야간")
# shift_rate = df.groupby("shift")["passorfail"].mean().reset_index()
# fig2 = px.bar(shift_rate, x="shift", y="passorfail", title="경문시간 불량")
# st.plotly_chart(fig2, use_container_width=True)

# # 요일 간 시간 불량 힐트맵
# st.subheader("3. 요일 x 시간 불량 히트맵")
# heat_df = df.pivot_table(index="hour", columns="weekday", values="passorfail", aggfunc="mean")
# heat_df = heat_df[weekday_order]
# fig3 = px.imshow(heat_df, color_continuous_scale="YlOrRd", labels=dict(color="불량률"), text_auto='.2f')
# fig3.update_layout(title="요일-시간 불량 히트맵", xaxis_title="요일", yaxis_title="시간")
# st.plotly_chart(fig3, use_container_width=True)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(layout="wide")

# 폰트 및 스타일 적용
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Noto Sans KR', sans-serif !important;
    }

    #section[data-testid="stSidebar"] {
    #    background-color: #384858;
    #}

    #section[data-testid="stSidebar"] * {
    #    color: white !important;
    #}

    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select {
        background-color: #f0f2f6 !important;
        color: #000000 !important;
        border-radius: 0.375rem;
    }

    section[data-testid="stSidebar"] svg {
        fill: black !important;
    }

    .card-inner {
        background-color: white;
        min-height: 100%;
        padding: 1rem;
        border-radius: 0px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .card-inner h5 {
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    </style>
""", unsafe_allow_html=True)

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv", low_memory=False)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()

    # ✅ 요일을 한글로 변환
    weekday_map = {
        "Monday": "월", "Tuesday": "화", "Wednesday": "수",
        "Thursday": "목", "Friday": "금", "Saturday": "토", "Sunday": "일"
    }
    df["weekday_kr"] = df["weekday"].map(weekday_map)

    df["shift"] = df["hour"].apply(lambda x: "주간" if 6 <= x < 18 else "야간")
    return df

df = load_data()

# 필터 처리
st.sidebar.header("필터 조건")
min_date = df["datetime"].min().date()
max_date = df["datetime"].max().date()
start_date = st.sidebar.date_input("시작 날짜", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("종료 날짜", min_value=min_date, max_value=max_date, value=max_date)
filtered_df = df[(df["datetime"].dt.date >= start_date) & (df["datetime"].dt.date <= end_date)]

mold_options = sorted(filtered_df["mold_code"].astype(str).unique())
selected_molds = st.sidebar.multiselect("금형 코드", mold_options, default=mold_options)
filtered_df = filtered_df[filtered_df["mold_code"].astype(str).isin(selected_molds)]

if len(filtered_df) == 0:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 1.5rem;'>
        <h1 style='flex: 1; text-align: center; margin: 0;'>과거 분석 데스크</h1>
        <h4 style='flex: 1; text-align: right; margin: 0; color: #666;'>총 샘플 수: {len(filtered_df):,}건</h4>
    </div>
""", unsafe_allow_html=True)

# 카드 구성
cards = [
    {"title": "요일별 불량률", "key": "weekday"},
    {"title": "주간 vs 야간 불량률", "key": "shift"},
    {"title": "요일 x 시간 히트맵", "key": "heat"},
    {"title": "생산건수 별 불량률", "key": "bubble"},
    {"title": "불량률 추이", "key": "trend"},
    {"title": "양품 vs 불량 변수 분포", "key": "distribution"}
]

for i in range(0, len(cards), 3):
    cols = st.columns([1, 1, 1], gap="small")
    for j, card in enumerate(cards[i:i+3]):
        with cols[j]:
            with st.container():
                st.markdown(f'''
                    <div class="card-inner">
                        <h5><strong>{card['title']}</strong></h5>
                        <div class="graph-wrapper">
    ''', unsafe_allow_html=True)

                fig = None

            if card["key"] == "weekday":
                order_kr = ["월", "화", "수", "목", "금", "토", "일"]
                data = filtered_df.groupby("weekday_kr")["passorfail"].mean().reindex(order_kr).reset_index()
                fig = px.bar(
                    data,
                    x="weekday_kr",
                    y="passorfail",
                    height=250,
                    labels={"passorfail": "불량률"},
                    color_discrete_sequence=["#5B62F6"]
                )
                fig.update_layout(yaxis_tickangle=0)

            elif card["key"] == "shift":
                data = filtered_df.groupby("shift")["passorfail"].mean().reset_index()
                fig = px.bar(
                    data,
                    x="shift",
                    y="passorfail",
                    height=250,
                    labels={"passorfail": "불량률"},
                    color="shift",
                    color_discrete_map={"주간": "#71E5E4", "야간": "#71A2EE"}
                )
                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis_tickangle=0
                )

            elif card["key"] == "heat":
                order_kr = ["월", "화", "수", "목", "금", "토", "일"]
                heat_df = filtered_df.pivot_table(index="hour", columns="weekday_kr", values="passorfail", aggfunc="mean")
                heat_df = heat_df[order_kr]
                fig = px.imshow(
                    heat_df,
                    text_auto=".2f",
                    color_continuous_scale=["#ffffff", "#5B62F6"],
                    height=250
                )
                fig.update_layout(
                    xaxis_title="요일",
                    yaxis_title="시간",
                    yaxis_tickangle=0
                )
                fig.update_yaxes(autorange="reversed")

            elif card["key"] == "bubble":
                agg = filtered_df.groupby("mold_code").agg(
                    fail_rate=("passorfail", "mean"),
                    count=("passorfail", "count")
                ).reset_index()
                fig = px.scatter(
                    agg,
                    x="count", y="fail_rate", size="count", color="fail_rate",
                    hover_name="mold_code",
                    color_continuous_scale=[[0, "#BACBFA"], [1, "#444EED"]],
                    labels={"count": "생산건수", "fail_rate": "불량률"},
                    height=250
                )
                fig.update_layout(yaxis_tickangle=0)

            elif card["key"] == "trend":
                data = filtered_df.resample("D", on="datetime")["passorfail"].mean().reset_index()
                fig = px.line(
                    data,
                    x="datetime", y="passorfail",
                    markers=True,
                    height=250,
                    labels={"passorfail": "불량률"},
                    color_discrete_sequence=["#B66EEB"]
                )
                fig.update_layout(yaxis_tickangle=0)

            elif card["key"] == "distribution":
                numeric_cols = filtered_df.select_dtypes(include="number").columns
                numeric_cols = [col for col in numeric_cols if col not in ["passorfail", "hour", "id"]]
                default_var = numeric_cols[0] if numeric_cols else None
                var = st.session_state.get(f"select_{card['key']}", default_var)

                good = filtered_df[filtered_df["passorfail"] == 0]
                bad = filtered_df[filtered_df["passorfail"] == 1]

                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=good[var],
                    name="양품",
                    marker_color="rgba(146,180,225,0.5)",
                    nbinsx=40,
                    opacity=0.5
                ))
                fig.add_trace(go.Histogram(
                    x=bad[var],
                    name="불량",
                    marker_color="#6EA7E8",
                    nbinsx=40,
                    opacity=1.0
                ))
                fig.update_layout(
                    barmode='overlay',
                    height=250,
                    legend_title_text="불량 여부",
                    xaxis_title=var,
                    yaxis_title="빈도",
                    yaxis_tickangle=0
                )

            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{card['key']}")

            if card["key"] == "distribution":
                var = st.selectbox("변수 선택", numeric_cols, key=f"select_{card['key']}")

                st.markdown("</div>", unsafe_allow_html=True)
