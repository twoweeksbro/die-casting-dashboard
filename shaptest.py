import streamlit as st
import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

st.set_page_config("SHAP 텍스트 출력", layout="wide")
st.title("🎯 SHAP 변수 영향도 Top 3")

# 데이터 로딩 및 모델 학습
@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return X, y, model

X, y, model = load_data()

# SHAP 계산은 전체 데이터로 다시
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 샘플 선택
idx = st.slider("샘플 인덱스 선택", 0, len(X)-1, 0)
sample = X.iloc[[idx]]

# 예측 결과
pred = model.predict(sample)[0]
proba = model.predict_proba(sample)[0][1]
st.write(f"📌 예측 결과: `{pred}` | 확률: `{proba:.2f}` | 실제값: `{y[idx]}`")

# SHAP 상위 변수 출력
shap_val = shap_values[1][idx]  # 클래스 1 기준
feature_names = X.columns.to_numpy()
top_idx = np.argsort(np.abs(shap_val))[::-1][:3]

st.subheader("🔍 예측에 가장 크게 기여한 변수 Top 3")
for i in top_idx:
    st.markdown(f"- **{feature_names[i]}**: 영향도 = `{shap_val[i]:.4f}`")
