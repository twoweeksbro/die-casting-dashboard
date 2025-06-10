import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from catboost import Pool
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report
import io


# 페이지 설정
st.set_page_config(layout="wide")
st.title("불량 확률 예측 시뮬레이터")


def display_model_metrics_card():
    import io
    df = load_data()
    model = load_model()

    # --- 성능 데이터 준비 ---
    X = df.drop(columns=['id', 'passorfail', 'registration_time', 'heating_furnace'])
    y = df['passorfail']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=1449, stratify=y
    )
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')
    report = classification_report(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    # --- 카드 스타일 ---
    card_style = """
    <style>
    .card {background:#f9f9f9;border-radius:8px;padding:12px;margin:6px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}
    .card-title{font-weight:bold;font-size:1rem;}
    .card-value{font-size:0.9rem;
        white-space: normal;
        word-break: break-all;
        overflow-wrap: break-word;
        }
    .metrics-card {background:#f9f9f9;border-radius:8px;padding:16px;margin:6px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}
    .metrics-title{font-weight:bold;font-size:1.2rem;margin-bottom:8px;}
    .metrics-value{font-size:1.8rem;font-weight:bold;color:#2c3e50;}
    .report-card {background:#f9f9f9;border-radius:8px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,0.1);}
    .report-title{font-weight:bold;font-size:1rem;margin-bottom:8px;}
    .mono{font-family:monospace;font-size:0.85rem;background:#ffffff;padding:12px;border-radius:4px;border:1px solid #e0e0e0;}
    </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    st.markdown("---")  # 구분선
    st.markdown("## 모델 성능")
    
    # F1 Score와 Recall 카드
    st.subheader("성능 지표")
    metric_cols = st.columns(2)
    
    with metric_cols[0]:
        st.markdown(f"""
        <div class='metrics-card'>
          <div class='metrics-title'>F1 Score (Macro)</div>
          <div class='metrics-value'>{f1:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown(f"""
        <div class='metrics-card'>
          <div class='metrics-title'>Recall (Macro)</div>
          <div class='metrics-value'>{recall:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Confusion Matrix - 예쁜 카드 형태로
    st.subheader("Confusion Matrix")
    
    # 실제 confusion matrix 값 사용
    cm_values = [[14009, 56], [61, 594]]
    
    # 4개의 카드로 confusion matrix 표현
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 10px;'>
            <strong style='font-size: 1.1rem; color: #2c3e50;'>실제 정상 (Normal)</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # True Negative (정상을 정상으로)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4CAF50, #45a049); 
                    color: white; padding: 20px; border-radius: 10px; 
                    text-align: center; margin-bottom: 10px;
                    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);'>
            <div style='font-size: 0.9rem; opacity: 0.9;'>정상 → 정상 (True Negative)</div>
            <div style='font-size: 2rem; font-weight: bold; margin-top: 5px;'>{cm_values[0][0]:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # False Negative (불량을 정상으로)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #FF9800, #F57C00); 
                    color: white; padding: 20px; border-radius: 10px; 
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(255, 152, 0, 0.3);'>
            <div style='font-size: 0.9rem; opacity: 0.9;'>불량 → 정상 (False Negative)</div>
            <div style='font-size: 2rem; font-weight: bold; margin-top: 5px;'>{cm_values[1][0]:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 10px;'>
            <strong style='font-size: 1.1rem; color: #2c3e50;'>실제 불량 (Defect)</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # False Positive (정상을 불량으로)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #FF5722, #D84315); 
                    color: white; padding: 20px; border-radius: 10px; 
                    text-align: center; margin-bottom: 10px;
                    box-shadow: 0 4px 8px rgba(255, 87, 34, 0.3);'>
            <div style='font-size: 0.9rem; opacity: 0.9;'>정상 → 불량 (False Positive)</div>
            <div style='font-size: 2rem; font-weight: bold; margin-top: 5px;'>{cm_values[0][1]:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # True Positive (불량을 불량으로)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #2196F3, #1976D2); 
                    color: white; padding: 20px; border-radius: 10px; 
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(33, 150, 243, 0.3);'>
            <div style='font-size: 0.9rem; opacity: 0.9;'>불량 → 불량 (True Positive)</div>
            <div style='font-size: 2rem; font-weight: bold; margin-top: 5px;'>{cm_values[1][1]:,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 정확도 요약
    total = sum(sum(row) for row in cm_values)
    accuracy = (cm_values[0][0] + cm_values[1][1]) / total
    
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; 
                text-align: center; margin-top: 15px; border: 1px solid #e9ecef;'>
        <strong style='color: #2c3e50;'>전체 정확도: {accuracy:.1%} ({cm_values[0][0] + cm_values[1][1]:,}/{total:,})</strong>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 캐싱된 모델 로드 및 데이터 로드 함수들
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str = "catboost_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

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

@st.cache_resource
def load_params(path: str = "best_params_BS_check.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

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

# ─────────────────────────────────────────────────────────────────────────────
# 메인 예측 섹션 (상단 배치)
# ─────────────────────────────────────────────────────────────────────────────
def main_prediction():
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
        "생산 상태 및 장비 조건": ["working", "tryshot_signal", 'mold_code'],
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

    # st.sidebar.header("입력 변수 선택")
    # selected_vars = st.sidebar.multiselect(
    #     "모델 입력 변수 선택", feature_names, default=feature_names
    # )

    selected_vars = feature_names

    st.header("입력값 설정")
    cols_layout = st.columns(len(group_dict))
    input_data = {}

    # 입력값 설정
    # for idx, (group, vars_) in enumerate(group_dict.items()):
    #     with cols_layout[idx]:
    #         st.subheader(group)
    #         for var in vars_:
    #             if var not in selected_vars:
    #                 continue
    #             if var in cat_feature_names:
    #                 options = sorted(df[var].astype(str).unique())
    #                 input_data[var] = st.selectbox(var, options)
    #             else:
    #                 # 분위수 기반 범위 계산
    #                 q_low, q_high = df[var].quantile([0.025, 0.975]).tolist()
    #                 # q_low, q_high = df[var].quantile([0.1, 0.9]).tolist()
    #                 # 범위가 유효한지를 확인
    #                 if q_high <= q_low:
    #                     # 범위가 없거나 동일한 경우 number_input 사용
    #                     if pd.api.types.is_integer_dtype(df[var]):
    #                         default_v = int(q_low)
    #                     else:
    #                         default_v = float(q_low)
    #                     input_data[var] = st.number_input(
    #                         label=var,
    #                         value=default_v
    #                     )
    #                 else:
    #                     # 슬라이더 사용: 기본값을 중앙으로 설정
    #                     if pd.api.types.is_integer_dtype(df[var]):
    #                         min_v = int(q_low)
    #                         max_v = int(q_high)
    #                         default_v = int((min_v + max_v) / 2)
    #                         step = 1
    #                         input_data[var] = st.slider(
    #                             label=var,
    #                             min_value=min_v,
    #                             max_value=max_v,
    #                             value=default_v,
    #                             step=step
    #                         )
    #                     else:
    #                         min_v = float(q_low)
    #                         max_v = float(q_high)
    #                         default_v = float((min_v + max_v) / 2.0)
    #                         raw_step = (max_v - min_v) / 100.0
    #                         if raw_step <= 0:
    #                             raw_step = 1e-6
    #                         step = float(raw_step)
    #                         input_data[var] = st.slider(
    #                             label=var,
    #                             min_value=min_v,
    #                             max_value=max_v,
    #                             value=default_v,
    #                             step=step
    #                         )
    
    initial_values = {
    'upper_mold_temp1': 212,
    'sleeve_temperature': 517,
    'lower_mold_temp1': 256.27,
    'lower_mold_temp2': 232.70,
    'cast_pressure' : 307.67
    }
    
    for idx, (group, vars_) in enumerate(group_dict.items()):
        with cols_layout[idx]:
            st.subheader(group)
            for var in vars_:
                if var not in selected_vars:
                    continue
                
                # if var == "working":
                #     # 값이 1/0(str) 형태일 때만 토글
                #     current = df[var].mode()[0]
                #     val = st.toggle("가동 여부 (working)", value=(current == "1"))
                #     input_data[var] = "1" if val else "0"
                # elif var == "tryshot_signal":
                #     current = df[var].mode()[0]
                #     val = st.toggle("트라이샷 신호 (tryshot_signal)", value=(current == "1"))
                #     input_data[var] = "1" if val else "0"
                # # mold_code: selectbox
                # elif var == "mold_code":
                #     options = sorted(df[var].unique())
                #     input_data[var] = st.selectbox("금형 코드 (mold_code)", options)

                
                # 초기값을 가져오거나, 없으면 None (또는 기본값 로직 사용)
                # if var in initial_values:
                #     default_value_for_input = initial_values[var]
                # else:
                #     default_value_for_input = None # 각 위젯의 기존 기본값 로직을 사용

                if var in cat_feature_names:
                    options = sorted(df[var].astype(str).unique())
                    
                   

                    # st.selectbox 초기값 설정
                    if var in initial_values and initial_values[var] in options:
                        default_index = options.index(initial_values[var])
                        input_data[var] = st.selectbox(var, options, index=default_index)
                    else: # 초기값이 없거나 유효하지 않은 경우, 기존 로직대로
                        input_data[var] = st.selectbox(var, options)

                else: # 수치형 변수
                    # 분위수 기반 범위 계산
                    q_low, q_high = df[var].quantile([0.025, 0.975]).tolist()
                    
                    # 기본값 설정 로직 (기존 코드에서 default_v 계산)
                    current_default_v = float((q_low + q_high) / 2.0)
                    if pd.api.types.is_integer_dtype(df[var]):
                        current_default_v = int(current_default_v)


                    # 특정 컬럼에 대한 초기값이 있다면 그것을 사용
                    if var in initial_values:
                        # 초기값이 min_value와 max_value 범위 안에 있는지 확인하여 유효성 보장
                        if q_high > q_low: # 슬라이더 범위가 유효할 때
                            if isinstance(initial_values[var], (int, float)):
                                default_value_for_input = max(float(q_low), min(float(q_high), float(initial_values[var])))
                            else: # 초기값이 숫자가 아니면 기존 기본값 사용
                                default_value_for_input = current_default_v
                        else: # number_input 사용해야 할 때 (범위가 없거나 동일)
                            if isinstance(initial_values[var], (int, float)):
                                default_value_for_input = float(initial_values[var])
                            else: # 초기값이 숫자가 아니면 기존 기본값 사용
                                default_value_for_input = current_default_v
                    else:
                        default_value_for_input = current_default_v


                    # 범위가 유효한지를 확인 (q_high <= q_low 인 경우 number_input 사용)
                    if q_high <= q_low:
                        # number_input 사용
                        
                        if var == "working":
                        # 값이 1/0(str) 형태일 때만 토글
                            current = df[var].mode()[0]
                            val = st.toggle("가동 여부 (working)", value=True,key='working',)
                            input_data[var] = "1" if val else "0"
                        elif var == "tryshot_signal":
                            current = df[var].mode()[0]
                            val = st.toggle("트라이샷 신호 (tryshot_signal)", value=(current == "1"), key='tryshot_signal')
                            input_data[var] = "1" if val else "0"
                        # mold_code: selectbox
                        elif var == "mold_code":
                            options = sorted(df[var].unique())
                            input_data[var] = st.selectbox("금형 코드 (mold_code)", options)
                            
                        # if pd.api.types.is_integer_dtype(df[var]):
                        #     input_data[var] = st.number_input(
                        #         label=var,
                        #         value=int(default_value_for_input) # 정수형
                        #     )
                        # else:
                        #     input_data[var] = st.number_input(
                        #         label=var,
                        #         value=float(default_value_for_input) # 실수형
                        #     )
                    else:
                        # 슬라이더 사용
                        if pd.api.types.is_integer_dtype(df[var]):
                            min_v = int(q_low)
                            max_v = int(q_high)
                            step = 1
                            input_data[var] = st.slider(
                                label=var,
                                min_value=min_v,
                                max_value=max_v,
                                value=int(default_value_for_input), # 정수형
                                step=step
                            )
                        else:
                            min_v = float(q_low)
                            max_v = float(q_high)
                            raw_step = (max_v - min_v) / 100.0
                            if raw_step <= 0:
                                raw_step = 1e-6
                            step = float(raw_step)
                            input_data[var] = st.slider(
                                label=var,
                                min_value=min_v,
                                max_value=max_v,
                                value=float(default_value_for_input), # 실수형
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

# ─────────────────────────────────────────────────────────────────────────────
# 환경 정보 및 모델 정보 섹션 (하단 배치)
# ─────────────────────────────────────────────────────────────────────────────
def display_system_info():
    st.markdown("---")  # 구분선
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

    # 시스템 정보 딕셔너리
    sys_info = {
        "CPU": cpu_line,
        "RAM Total (GiB)": ram_total,
        "GPU": gpu_name,
        "GPU Memory": gpu_mem
    }

    # 카드 스타일
    card_style = """
    <style>
    .card {background:#f9f9f9;border-radius:8px;padding:12px;margin:6px;box-shadow:0 1px 3px rgba(0,0,0,0.1);}
    .card-title{font-weight:bold;font-size:1rem;}
    .card-value{font-size:0.9rem;
        white-space: normal;
        word-break: break-all;
        overflow-wrap: break-word;
        }
    </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)

    # 시스템 정보 카드 출력
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

    # 라이브러리 버전 정보
    lib_info = {
        "Python": "3.11.13",
        "catboost": "1.2.8",
        "numpy": "2.0.2",
        "pandas": "2.2.2",
        "sklearn": "1.6.1"
    }

    st.subheader("라이브러리 버전")
    items = list(lib_info.items())
    rows = [items[:3], items[3:6]]

    for row in rows:
        cols = st.columns(3)
        for idx, (pkg, ver) in enumerate(row):
            with cols[idx]:
                st.markdown(f"""
                <div class='card'>
                  <div class='card-title'>{pkg}</div>
                  <div class='card-value'>{ver}</div>
                </div>
                """, unsafe_allow_html=True)

def display_model_info():
    # 모델 정보 섹션
    version = "diecasting_catboost_borderline_smote_v4.2.16_20250610"

    

    st.subheader("모델 정보")
    st.caption(version)
    # 아래에 CatBoost + BorderlineSMOTE 사용 내용 추가
    st.markdown("""
    <div style='margin-top: 18px; padding: 15px 18px; background: #f1f7fc; border-radius: 8px; border: 1px solid #e0ecf7; color: #12518a; font-size: 1.01rem;'>
      <b>모델 설명</b><br>
      - <b>CatBoost</b> 분류기를 기반,<br>
      - <b>BorderlineSMOTE</b> 오버샘플링 기법을 적용하여 클래스 불균형을 보정
    </div>
    """, unsafe_allow_html=True)
    #st.markdown("학습에 사용된 CatBoost 모델의 하이퍼파라미터 및 간단 설명입니다.")
    
    # 모델 파라미터 로드
    try:
        params = load_params()
        items = list(params.items())
        # 3열씩 카드로 나누기
        rows = [items[i:i+3] for i in range(0, len(items), 3)]
        for row in rows:
            cols = st.columns(3)
            for idx, (key, value) in enumerate(row):
                with cols[idx]:
                    st.markdown(f"""
                    <div class='card'>
                      <div class='card-title'>{key}</div>
                      <div class='card-value'>{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"모델 정보 로드 중 오류가 발생했습니다: {e}")
    
    

        


# 메인 실행
if __name__ == "__main__":
    # 1. 예측 부분 (상단)
    main_prediction()
    
    # 2. 환경 정보 및 모델 정보 (하단)
    display_system_info()
    display_model_info()
    display_model_metrics_card()

#X.isnull().sum()  # 결측치 확인