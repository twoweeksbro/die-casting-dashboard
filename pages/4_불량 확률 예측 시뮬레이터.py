import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from catboost import Pool

# 페이지 설정
st.set_page_config(layout="wide")
st.title("불량 확률 예측 시뮬레이터")

# ─────────────────────────────────────────────────────────────────────────────
# 모델 파라미터 로드 및 설명 표시
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_params(path: str = "best_params_BS_check.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

# 사전 로드하여 화면에 표시
params = load_params()


# ─────────────────────────────────────────────────────────────────────────────
# 환경 정보 섹션 (카드 UI: 파일 기반)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 환경 정보")

# system_info.txt 읽기
try:
    with open("system_info.txt", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
except FileNotFoundError:
    st.error("system_info.txt 파일을 찾을 수 없습니다.")
    lines = []

# 블록 추출 함수
def extract_block(lines, start_marker, end_marker=None):
    if start_marker not in lines:
        return []
    start = lines.index(start_marker) + 1
    if end_marker and end_marker in lines[start:]:
        end = lines.index(end_marker, start)
    else:
        end = len(lines)
    return [ln.strip() for ln in lines[start:end] if ln.strip()]

# Python
py_block = extract_block(lines, "-- Python Version --", "-- Installed Packages --")
python_version = py_block[0] if py_block else "N/A"

# CPU
cpu_block = extract_block(lines, "-- CPU Info --", "-- Memory Info --")
cpu_line = cpu_block[0].split(":",1)[1].strip() if cpu_block else "N/A"

# RAM Total
mem_block = extract_block(lines, "-- Memory Info --", "-- GPU Info --")
ram_total = "N/A"
for ln in mem_block:
    if ln.startswith("Mem:"):
        parts = ln.split()
        raw = parts[1] if len(parts)>=2 else ""
        try:
            ram_total = f"{float(raw.replace('Gi','')):.1f}"
        except:
            pass
        break

# GPU Name + Memory
gpu_block = extract_block(lines, "-- GPU Info --")
gpu_name = gpu_mem = "N/A"
for i, ln in enumerate(gpu_block):
    if "NVIDIA" in ln and "Off" in ln:
        parts = ln.split()
        start = parts.index("NVIDIA")
        # 이름 합치기
        name_parts = []
        for p in parts[start+1:]:
            if p=="Off" or p.startswith("|"):
                break
            name_parts.append(p)
        gpu_name = "NVIDIA " + " ".join(name_parts)
        # 다음 줄에서 total 메모리
        if i+1 < len(gpu_block) and "MiB /" in gpu_block[i+1]:
            seg = gpu_block[i+1].split("|")[2].strip()
            gpu_mem = seg.split("/")[1].strip()
        break

# 시스템 정보 딕셔너리에 모두 모아두기
sys_info = {
    "CPU": cpu_line,
    "RAM Total (GiB)": ram_total,
    "GPU": gpu_name,
    "GPU Memory": gpu_mem
}




# ─────────────────────────────────────────────────────────────────────────────
# 카드 스타일 (중복 선언 없이 한 번만)
# ─────────────────────────────────────────────────────────────────────────────
card_style = """
<style>
.card {background:#f9f9f9;border-radius:8px;padding:12px;margin:6px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}
.card-title{font-weight:bold;font-size:1rem;}
.card-value{font-size:0.9rem;
    white-space: normal;              /* 기본적으로 줄바꿈이 가능하도록 */
    word-break: break-all;            /* 단어 중간에서도 줄바꿈 허용 */
    overflow-wrap: break-word;        /* 긴 단어나 연속 텍스트도 적절히 줄바꿈 */
    }
</style>
"""
st.markdown(card_style, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 시스템 정보 카드 출력
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("시스템 정보")
cols = st.columns(len(sys_info))
for idx, (k, v) in enumerate(sys_info.items()):
    with cols[idx]:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>{k}</div>
          <div class='card-value'>{v}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 라이브러리 버전 카드 출력 (2행×3열)
# ─────────────────────────────────────────────────────────────────────────────
# 예시 lib_info: Python 버전 + 4개 패키지
lib_info = {
    "Python": "3.11.13",
    "catboost": "1.2.8",
    "numpy": "2.0.2",
    "pandas": "2.2.2",
    "sklearn": "1.6.1"
}

# 카드 CSS 정의 (앱 최상단에 한 번만 선언)
card_style = """
<style>
.card {background:#f9f9f9;border-radius:8px;padding:12px;margin:6px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}
.card-title{font-weight:bold;font-size:1rem;}
.card-value{font-size:0.9rem;}
</style>
"""
st.markdown(card_style, unsafe_allow_html=True)

# "라이브러리 버전" 섹션
st.subheader("라이브러리 버전")

items = list(lib_info.items())  # 순서대로 5개 튜플 리스트

# 1행: 첫 3개, 2행: 나머지 (2개)
rows = [items[:3], items[3:6]]  # items[3:6]은 실제로 2개만 들어있음

for row in rows:
    cols = st.columns(3)
    for idx, (pkg, ver) in enumerate(row):
        # idx가 0,1,2 중 하나이므로 두 번째 행에서는 idx가 0,1까지만 반복됨.
        with cols[idx]:
            st.markdown(f"""
            <div class='card'>
              <div class='card-title'>{pkg}</div>
              <div class='card-value'>{ver}</div>
            </div>
            """, unsafe_allow_html=True)
    # row에 3개가 안 되는 경우(cols[idx] 참조는 idx만큼만 됨), 나머지 컬럼은 비어 있음.


# ─────────────────────────────────────────────────────────────────────────────
# 모델 정보 섹션 추가: 환경 정보 바로 아래
# ─────────────────────────────────────────────────────────────────────────────
version = "diecasting_catboost_borderline_smote_v4.2.16_20250610"

st.subheader("모델 정보")
st.caption(version)

st.markdown("학습에 사용된 CatBoost 모델의 하이퍼파라미터 및 간단 설명입니다.")
# params: load_params()로 불러온 dict
try:
    items = list(params.items())
    # 3열씩 카드로 나누기
    rows = [items[i:i+3] for i in range(0, len(items), 3)]
    for row in rows:
        cols = st.columns(3)
        for idx, (key, value) in enumerate(row):
            with cols[idx]:
                # value가 리스트/튜플/딕셔너리 등 복합형일 경우 str()로 변환
                st.markdown(f"""
                <div class='card'>
                  <div class='card-title'>{key}</div>
                  <div class='card-value'>{value}</div>
                </div>
                """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"모델 정보 로드 중 오류가 발생했습니다: {e}")


# 캐싱된 모델 로드
@st.cache_resource
def load_model(path: str = "catboost_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

# 캐싱된 데이터 로드 및 전처리
@st.cache_data
def load_data(path: str = "data/train_v1.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # working, tryshot_signal 매핑
    df['working'] = df['working'].map({'가동': 1, '정지': 0})
    df['tryshot_signal'] = df['tryshot_signal'].fillna('N').map({'Y': 1, 'N': 0})
    df['mold_code'] = df['mold_code'].astype(str)
    # 문자열로 변환 후 라벨 인코딩
    for col in ['working', 'tryshot_signal']:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# 예측 수행 함수
def predict_defect(model, input_df: pd.DataFrame, feature_names: list,
                   cat_feature_names: list, df: pd.DataFrame) -> float:
    # 누락된 변수 채우고 순서 맞춤
    for col in feature_names:
        if col not in input_df.columns:
            if col in df.select_dtypes(include=['number']).columns:
                input_df[col] = df[col].mean()
            else:
                input_df[col] = df[col].mode()[0]
    input_df = input_df[feature_names]
    # Pool 생성
    pool = Pool(input_df, cat_features=cat_feature_names)
    return model.predict_proba(pool)[0][1]

# 결과 경고 표시
def display_warning(prob: float):
    if prob >= 0.7:
        color, message = "#ffcccc", "불량 확률이 매우 높습니다! 주요 공정을 점검하세요."
    elif prob >= 0.4:
        color, message = "#fff3cd", "불량 위험이 다소 있습니다. 주의가 필요합니다."
    else:
        color, message = "#d4edda", "불량 확률이 낮습니다. 공정이 안정적입니다."
    st.markdown(
        f"""
        <div style='background-color:{color}; padding:1rem; border-radius:10px;'>
            <strong>{message}</strong>
        </div>
        """, unsafe_allow_html=True)

# 메인 함수
def main():
    model = load_model()
    df = load_data()

    # 피처 목록
    drop_cols = ["id", "passorfail", "registration_time", "heating_furnace"]
    feature_names = [c for c in df.columns if c not in drop_cols]
    # 모델 학습 시 지정된 카테고리 피처 인덱스 및 이름
    cat_indices = model.get_cat_feature_indices()
    cat_feature_names = [feature_names[i] for i in cat_indices]

    # UI용 그룹
    group_dict = {
        "생산 상태 및 장비 조건": ["working", "tryshot_signal"],
        "온도 관련": [
            "upper_mold_temp1", "upper_mold_temp2", "molten_temp",
            "sleeve_temperature", "Coolant_temperature",
            "lower_mold_temp1", "lower_mold_temp2"
        ],
        "성형 공정 (속도/압력/두께)": [
            "low_section_speed", "high_section_speed", "cast_pressure",
            "molten_volume", "biscuit_thickness", "physical_strength"
        ],
    }

    st.sidebar.header("입력 변수 선택")
    selected_vars = st.sidebar.multiselect(
        "모델 입력 변수 선택", feature_names, default=feature_names
    )

    st.header("입력값 설정")
    cols_layout = st.columns(len(group_dict))
    input_data = {}

    # 입력값 설정 부분 수정 예시: 기본값을 중앙에 위치시키기
    for idx, (group, vars_) in enumerate(group_dict.items()):
        with cols_layout[idx]:
            st.subheader(group)
            for var in vars_:
                if var not in selected_vars:
                    continue
                if var in cat_feature_names:
                    options = sorted(df[var].astype(str).unique())
                    input_data[var] = st.selectbox(var, options)
                else:
                    # 분위수 기반 범위 계산
                    q_low, q_high = df[var].quantile([0.025, 0.975]).tolist()
                    # 범위가 유효한지를 확인
                    if q_high <= q_low:
                        # 범위가 없거나 동일한 경우 number_input 사용. 기본값은 해당 고정값(q_low) 또는 중앙(같으므로 동일)
                        # 정수형이면 int, float면 float
                        if pd.api.types.is_integer_dtype(df[var]):
                            default_v = int(q_low)
                        else:
                            default_v = float(q_low)
                        input_data[var] = st.number_input(
                            label=var,
                            value=default_v
                        )
                    else:
                        # 슬라이더 사용: 기본값을 중앙((q_low+q_high)/2)으로 설정
                        # 정수형 변수인 경우
                        if pd.api.types.is_integer_dtype(df[var]):
                            min_v = int(q_low)
                            max_v = int(q_high)
                            # 중앙값을 int로: (min+max)//2 또는 반올림할 수도 있음
                            default_v = int((min_v + max_v) / 2)
                            # 스텝은 1로 설정하는 것이 일반적
                            step = 1
                            input_data[var] = st.slider(
                                label=var,
                                min_value=min_v,
                                max_value=max_v,
                                value=default_v,
                                step=step
                            )
                        else:
                            # float형 변수
                            min_v = float(q_low)
                            max_v = float(q_high)
                            default_v = float((min_v + max_v) / 2.0)
                            # 슬라이더 스텝: 범위를 100단계로 나눈 값, 너무 작으면 최소값 지정
                            raw_step = (max_v - min_v) / 100.0
                            if raw_step <= 0:
                                raw_step = 1e-6
                            step = float(raw_step)
                            input_data[var] = st.slider(
                                label=var,
                                min_value=min_v,
                                max_value=max_v,
                                value=default_v,
                                step=step
                            )

    if st.button("불량 확률 예측하기"):
        input_df = pd.DataFrame([input_data])
        prob = predict_defect(
            model, input_df, selected_vars,
            [c for c in cat_feature_names if c in selected_vars], df
        )
        st.subheader("예측된 불량 확률")
        st.metric(label="불량 확률", value=f"{prob*100:.2f}%")
        display_warning(prob)

if __name__ == "__main__":
    main()

#X.isnull().sum()  # 결측치 확인