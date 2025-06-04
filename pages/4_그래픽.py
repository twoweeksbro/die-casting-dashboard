import plotly.graph_objects as go
import streamlit as st

# 공정 단계 텍스트
steps = ["금형 클램프", "용탕 주입", "고압 주입", "냉각", "금형 개방", "제품 배출", "검사"]
x_pos = [0, 1.5, 3, 4.5, 6, 7.5, 9]  # 위치 조정

# 실시간 공정 상태 (예: 예측 불량 확률 기준 색상)
colors = ["lightgreen", "lightgreen", "yellow", "red", "lightgreen", "lightgreen", "green"]
status = ["정상", "정상", "주의", "이상", "정상", "정상", "정상"]

fig = go.Figure()

# 각 공정 단계 시각화
for i, (step, x, color, stat) in enumerate(zip(steps, x_pos, colors, status)):
    fig.add_shape(type="rect", x0=x, x1=x+1, y0=0, y1=1,
                  line=dict(color="black"),
                  fillcolor=color)
    fig.add_trace(go.Scatter(
        x=[x + 0.5], y=[0.5],
        text=[f"{step}<br>{stat}"],
        mode="text",
        textfont=dict(size=14, color="black")
    ))

# 공정 흐름 화살표
for i in range(len(x_pos)-1):
    fig.add_annotation(
        x=x_pos[i+1]-0.25, y=0.5,
        ax=x_pos[i]+1.1, ay=0.5,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1
    )

fig.update_layout(
    width=1000, height=250,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    plot_bgcolor="white"
)

st.plotly_chart(fig, use_container_width=True)

# import plotly.graph_objects as go
# import streamlit as st

# st.subheader("📈 다이캐스팅 공정 흐름 시각화")

# fig = go.Figure()

# # 각 공정 단계 노드
# steps = ["금형 닫힘", "금속 주입", "금속 충진", "압력 유지", "금형 열림", "제품 배출"]

# # 단계 간 위치
# for i, step in enumerate(steps):
#     fig.add_trace(go.Scatter(
#         x=[i], y=[1], text=[step], mode="text+markers",
#         marker=dict(size=30, color="skyblue"), showlegend=False
#     ))

# # 단계 간 연결선
# for i in range(len(steps) - 1):
#     fig.add_annotation(
#         x=i + 1, y=1,
#         ax=i, ay=1,
#         xref="x", yref="y",
#         axref="x", ayref="y",
#         showarrow=True,
#         arrowhead=3,
#         arrowsize=1,
#         arrowwidth=2,
#         arrowcolor="gray"
#     )

# fig.update_layout(
#     xaxis=dict(visible=False),
#     yaxis=dict(visible=False),
#     height=200,
#     margin=dict(l=20, r=20, t=20, b=20)
# )

# st.plotly_chart(fig, use_container_width=True)