import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# n = np.array([61, 85, 75, 86, 64, 96, 87, 93, 67, 97, 77, 88, 90, 84, 65, 71, 69, 66, 98, 72])
# x_i = np.array([4, 3, 2, 4, 2, 4, 5, 3, 6, 7, 6, 5, 8, 5, 5, 3, 3, 4, 5, 8])
df = pd.read_csv("data/train.csv")
df.columns


# datetime 열이 datetime 형식이라고 가정
df['datetime'] = pd.to_datetime(df['registration_time'])

# '월-일-시' 형식으로 문자열 컬럼 생성
df['month_day_hour'] = df['datetime'].dt.strftime('%m-%d %H')

# 이제 그룹화 가능
grouped = df.groupby('month_day_hour').agg({'passorfail': 'mean'})
print(grouped.head())

# 월,일,시 기준
# 전체 불량율
p_hat = df['passorfail'].mean()
# 전체 개수
n = df.groupby('month_day_hour')['passorfail'].size().sort_index()
p_hat

ucl = p_hat + 3 * np.sqrt(p_hat * (1 - p_hat) / n)

lcl = p_hat - 3 * np.sqrt(p_hat * (1 - p_hat) / n)



day_i =grouped.index
p_i = grouped['passorfail']

res_df = pd.DataFrame({
    "Day": day_i,
    "Defective Rate": p_i,
    "UCL": ucl,
    "LCL": lcl,
    "Ave. Rate": [p_hat] * len(grouped)
})
# %matplotlib inline
res_df['Ave. Rate']
sns.lineplot(x="Day", y="Defective Rate", data=res_df[:50], marker="o", label="Defective Rate")
sns.lineplot(x="Day", y="UCL", data=res_df[:50], color='red', label="UCL")
sns.lineplot(x="Day", y="LCL", data=res_df[:50], color='red', label="LCL")
sns.lineplot(x="Day", y="Ave. Rate", data=res_df[:50], color='black', linestyle='--', label="Ave. Rate")
plt.fill_between(res_df[:50]["Day"], res_df[:50]["LCL"], res_df[:50]["UCL"], color='red', alpha=0.1)
plt.title('P chart')
plt.ylabel('Defective Rate')
plt.legend(loc="lower right")
plt.show()


sns.lineplot(x="Day", y="Defective Rate", data=res_df, marker="o", label="Defective Rate")
sns.lineplot(x="Day", y="UCL", data=res_df, color='red', label="UCL")
sns.lineplot(x="Day", y="LCL", data=res_df, color='red', label="LCL")
sns.lineplot(x="Day", y="Ave. Rate", data=res_df, color='black', linestyle='--', label="Ave. Rate")
plt.fill_between(res_df["Day"], res_df["LCL"], res_df["UCL"], color='red', alpha=0.1)
plt.title('P chart')
plt.ylabel('Defective Rate')
plt.legend(loc="lower right")
plt.show()







# 월,일 기준
# 전체 불량율
p_hat = df['passorfail'].mean()
# 전체 개수
n = df.groupby('time')['passorfail'].size().sort_index()
p_hat

ucl = p_hat + 3 * np.sqrt(p_hat * (1 - p_hat) / n)

lcl = p_hat - 3 * np.sqrt(p_hat * (1 - p_hat) / n)



day_i = df.value_counts('time').sort_index().index
p_i = df.groupby('time')['passorfail'].mean().sort_index()

res_df = pd.DataFrame({
    "Day": day_i,
    "Defective Rate": p_i,
    "UCL": ucl,
    "LCL": lcl,
    "Ave. Rate": [p_hat] * len(p_i)
})
# %matplotlib inline

sns.lineplot(x="Day", y="Defective Rate", data=res_df, marker="o", label="Defective Rate")
sns.lineplot(x="Day", y="UCL", data=res_df, color='red', label="UCL")
sns.lineplot(x="Day", y="LCL", data=res_df, color='red', label="LCL")
sns.lineplot(x="Day", y="Ave. Rate", data=res_df, color='black', linestyle='--', label="Ave. Rate")
plt.fill_between(res_df["Day"], res_df["LCL"], res_df["UCL"], color='red', alpha=0.1)
plt.title('P chart')
plt.ylabel('Defective Rate')
plt.legend(loc="lower right")
plt.show()

res_df.to_csv('mgmt.csv',index=False)



### 전체 불량율 대비 최근 불량율


df = pd.read_csv("data/train.csv")
df.columns


# datetime 열이 datetime 형식이라고 가정
df['datetime'] = pd.to_datetime(df['registration_time'])


# 누적 불량률 (전체 기준)

total_n = len(df)
total_defects = (df["passorfail"] == 1).sum()
p_hat = total_defects / total_n

# 실시간 추적용 리스트
n_recent = 1  # 최근 1개 (또는 10개 등으로 설정 가능)
df["rolling_defect_rate"] = df["passorfail"].rolling(n_recent).mean()

# UCL, LCL 계산 (p관리도 기준)
ucl = p_hat + 3 * np.sqrt(p_hat * (1 - p_hat) / n_recent)
lcl = p_hat - 3 * np.sqrt(p_hat * (1 - p_hat) / n_recent)
lcl = max(0, lcl)  # 음수 방지

import plotly.graph_objects as go

# 시각화
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["datetime"], y=df["rolling_defect_rate"], mode="lines+markers", name="실시간 불량률"))
fig.add_trace(go.Scatter(x=df["datetime"], y=[ucl]*len(df), line=dict(color="red", dash="dash"), name="UCL"))
fig.add_trace(go.Scatter(x=df["datetime"], y=[lcl]*len(df), line=dict(color="red", dash="dash"), name="LCL"))
fig.add_trace(go.Scatter(x=df["datetime"], y=[p_hat]*len(df), line=dict(color="black", dash="dot"), name="전체 평균"))

fig.update_layout(title="실시간 불량률 추적", yaxis_title="불량률", xaxis_title="시간")
fig.show()


######
# 불량률 최근 100개
import pandas as pd
import plotly.graph_objects as go

# 예시 DataFrame
# df = pd.read_csv("your_file.csv")
# 여기서 passorfail 컬럼이 존재한다고 가정

# 전체 누적 불량률
total_defect_rate = (df["passorfail"] == 1).mean()

# rolling window (100개씩 이동 평균)
rolling_defect_rate = df["passorfail"].rolling(window=100).mean()

# 시각화를 위한 인덱스 또는 시간축 (예: datetime이 있다면 x축으로 사용)
x = df.index  # 또는 df["datetime"]


fig = go.Figure()

# 전체 누적 불량률 (수평선)
fig.add_trace(go.Scatter(
    x=x,
    y=[total_defect_rate] * len(x),
    mode="lines",
    name="전체 누적 불량률",
    line=dict(color="black", dash="dot")
))

# 최근 100개 이동 불량률
fig.add_trace(go.Scatter(
    x=x,
    y=rolling_defect_rate,
    mode="lines",
    name="최근 100개 불량률",
    line=dict(color="red")
))

fig.update_layout(
    title="전체 vs 최근 100개 불량률 비교",
    xaxis_title="Index",
    yaxis_title="불량률",
    yaxis=dict(range=[0, 1]),
    template="plotly_white"
)

fig.show()




# 데이터 로드
df = pd.read_csv("./data/train.csv")

# 전체 누적 불량률
total_defect_rate = (df["passorfail"] == 1).mean()

# rolling window (100개 이동 평균)
rolling_defect_rate = df["passorfail"].rolling(window=100).mean()

# x축: 인덱스 또는 datetime
x = df.index  # 또는 df["datetime"]이 있다면 사용 가능

# 시각화
plt.figure(figsize=(12, 6))

# 전체 누적 불량률 (수평선)
plt.plot(x, [total_defect_rate] * len(x), linestyle='dotted', color='black', label='전체 누적 불량률')

# 최근 100개 이동 불량률
plt.plot(x, rolling_defect_rate, color='red', label='최근 100개 불량률')

plt.title("전체 vs 최근 100개 불량률 비교")
plt.xlabel("Index")
plt.ylabel("불량률")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()