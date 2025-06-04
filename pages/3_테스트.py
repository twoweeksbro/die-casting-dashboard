import streamlit as st
import plotly.graph_objects as go
import time
import random

placeholder = st.empty()

for i in range(100):
    simulated_prob = random.uniform(0, 100)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=simulated_prob,
        title={'text': "실시간 불량 확률"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    placeholder.plotly_chart(fig, use_container_width=True)
    time.sleep(1)



# import streamlit as st
# import plotly.graph_objects as go

# # Streamlit 슬라이더로 불량 확률 입력
# defect_prob = st.slider("불량 확률 (%)", 0, 100, 25)

# fig = go.Figure(go.Indicator(
#     mode="gauge+number",
#     value=defect_prob,
#     title={'text': "불량 확률"},
#     gauge={
#         'axis': {'range': [0, 100]},
#         'bar': {'color': "red"},
#         'steps': [
#             {'range': [0, 30], 'color': "lightgreen"},
#             {'range': [30, 70], 'color': "yellow"},
#             {'range': [70, 100], 'color': "red"}
#         ]
#     }
# ))

# st.plotly_chart(fig, use_container_width=True)
