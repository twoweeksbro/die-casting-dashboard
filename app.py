import streamlit as st

st.set_page_config(
    page_title="주조 공정 데이터 기반 모델 모니터링 대시보드",
    page_icon="",
    layout='wide'
)

# st.set_page_config(page_title="실시간 주조 공정 모니터링", page_icon="⏱️", layout="wide")



st.markdown("### 뭐든지 다해! **주조!** – 공정을 알고, 불량을 잡다.")

st.title('뭐든지 다해! 주조!')
st.subheader('공정을 알고, 불량을 잡다')
st.write("이 대시보드는 무엇이든 다 해줍니다.")

st.markdown("### 📂 페이지 바로가기")
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_실시간 공정 모니터링 대시보드.py", label="실시간 공정 모니터링")
    st.page_link("pages/1_실시간 공정 모니터링 대시보드.py", label="실시간 공정 모니터링")
    # st.page_link("pages/3_불량예측.py", label="🔮 불량 예측")

with col2:
    st.page_link("pages/2_과거 데이터 분석 대시보드.py", label="과거 데이터 분석")
    st.page_link("pages/2_과거 데이터 분석 대시보드.py", label="과거 데이터 분석")
    # st.page_link("pages/4_XAI_분석.py", label="🧠 XAI 해석")