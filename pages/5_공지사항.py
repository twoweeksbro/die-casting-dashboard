import streamlit as st
import plotly.graph_objects as go

st.set_page_config("Indicator 예시", layout="wide")
st.title("📊 Indicator 차트 예시")

sensor_data = {
    "용탕 온도 (°C)": 660.5,
    "주조 압력 (bar)": 120.7,
    "저속 구간 속도 (m/s)": 3.8
}

cols = st.columns(3)

for col, (label, value) in zip(cols, sensor_data.items()):
    with col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={'suffix': f" {label.split()[-1]}", 'font': {'size': 24}},
            title={'text': label, 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, value * 2], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': "mediumslateblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, value * 0.6], 'color': '#DFF5E3'},
                    {'range': [value * 0.6, value * 0.85], 'color': '#FFEAA7'},
                    {'range': [value * 0.85, value * 2], 'color': '#FFCCCC'}
                ]
            }
        ))
        fig.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=250)
        st.plotly_chart(fig, use_container_width=True)
