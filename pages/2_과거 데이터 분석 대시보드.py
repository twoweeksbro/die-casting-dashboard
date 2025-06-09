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

st.set_page_config(layout="wide")
st.title("과거 분석 데스크")

@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv", low_memory=False)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()
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

st.markdown(f"### 총 샘플 수: {len(filtered_df):,}건")

# 그래프 항목 정의
cards = [
    {"title": "요일별 불량률", "key": "weekday"},
    {"title": "주간 vs 야간 불량률", "key": "shift"},
    {"title": "요일 x 시간 히트맵", "key": "heat"},
    {"title": "생산건수 별 불량률", "key": "bubble"},
    {"title": "불량률 추이", "key": "trend"},
    {"title": "양품 vs 불량 변수 분포", "key": "distribution"}
]

if "expand_chart" not in st.session_state:
    st.session_state["expand_chart"] = {card["key"]: False for card in cards}

for i in range(0, len(cards), 3):
    cols = st.columns([1, 1, 1], gap="small")
    for j, card in enumerate(cards[i:i+3]):
        with cols[j]:
            st.markdown(f"**{card['title']}**")

            if card["key"] == "weekday":
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                data = filtered_df.groupby("weekday")["passorfail"].mean().reindex(order).reset_index()
                fig = px.bar(data, x="weekday", y="passorfail", height=250)

            elif card["key"] == "shift":
                data = filtered_df.groupby("shift")["passorfail"].mean().reset_index()
                fig = px.bar(data, x="shift", y="passorfail", height=250)

            elif card["key"] == "heat":
                heat_df = filtered_df.pivot_table(index="hour", columns="weekday", values="passorfail", aggfunc="mean")
                order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                heat_df = heat_df[order]
                fig = px.imshow(heat_df, text_auto=".2f", color_continuous_scale="Blues", height=250)

            elif card["key"] == "bubble":
                agg = filtered_df.groupby("mold_code").agg(
                    fail_rate=("passorfail", "mean"),
                    count=("passorfail", "count")
                ).reset_index()
                fig = px.scatter(
                    agg, x="count", y="fail_rate", size="count", color="fail_rate",
                    hover_name="mold_code", labels={"count": "생산건수", "fail_rate": "불량률"},
                    height=250
                )

            elif card["key"] == "trend":
                data = filtered_df.resample("D", on="datetime")["passorfail"].mean().reset_index()
                fig = px.line(data, x="datetime", y="passorfail", markers=True, height=250)

            elif card["key"] == "distribution":
                numeric_cols = filtered_df.select_dtypes(include="number").columns.drop(["passorfail", "hour"])
                var = st.selectbox("변수 선택", numeric_cols, key=f"select_{card['key']}")
                fig = px.histogram(
                    filtered_df,
                    x=var,
                    color=filtered_df["passorfail"].map({0: "양품", 1: "불량"}),
                    barmode="overlay",
                    nbins=40,
                    labels={"color": "불량 여부"},
                    height=250
                )

            st.plotly_chart(fig, use_container_width=True, key=f"mini_{card['key']}")

            if st.button(f"{card['title']} 전체 그래프 보기", key=f"btn_{card['key']}"):
                st.session_state["expand_chart"][card["key"]] = not st.session_state["expand_chart"].get(card["key"], False)

            if st.session_state["expand_chart"].get(card["key"], False):
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True, key=f"full_{card['key']}")
