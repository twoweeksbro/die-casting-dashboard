# pages/2_과거_분석_탐산.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("과거 데이터 분석 데스크")

# 데이터 로드
@st.cache_data

def load_data():
    df = pd.read_csv("data/train.csv")
    df["datetime"] = pd.to_datetime(df['time'] + " "+ df['date'])

    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()
    return df

df = load_data()

# 요일별 불량률
st.subheader("1. 요일별 불량")
fail_rate = df.groupby("weekday")["passorfail"].mean().reset_index()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
fig1 = px.bar(fail_rate, x="weekday", y="passorfail", title="요일별 불량", category_orders={"weekday": weekday_order})
st.plotly_chart(fig1, use_container_width=True)

# 주간 vs 야간 불량
st.subheader("2. 주간 vs 야간 불량")
df["shift"] = df["hour"].apply(lambda x: "주간" if 6 <= x < 18 else "야간")
shift_rate = df.groupby("shift")["passorfail"].mean().reset_index()
fig2 = px.bar(shift_rate, x="shift", y="passorfail", title="경문시간 불량")
st.plotly_chart(fig2, use_container_width=True)

# 요일 간 시간 불량 힐트맵
st.subheader("3. 요일 x 시간 불량 히트맵")
heat_df = df.pivot_table(index="hour", columns="weekday", values="passorfail", aggfunc="mean")
heat_df = heat_df[weekday_order]
fig3 = px.imshow(heat_df, color_continuous_scale="YlOrRd", labels=dict(color="불량률"), text_auto='.2f')
fig3.update_layout(title="요일-시간 불량 히트맵", xaxis_title="요일", yaxis_title="시간")
st.plotly_chart(fig3, use_container_width=True)
