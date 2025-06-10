import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import pickle

# 데이터 로딩
df = pd.read_csv("./data/train.csv")
X = df.drop(columns=['id', 'passorfail'])
y = df['passorfail']

X.columns
# 학습/검증 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=1449
)

# 수치형 / 범주형 컬럼 분리
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# 전처리 정의
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])


# 파이프라인 구성 및 학습
model = make_pipeline(
    preprocessor,
    RandomForestClassifier(n_estimators=100, random_state=42)
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
(y_pred==y_test).mean()




# 모델 저장
with open("model_rf.pkl", "wb") as f:
    pickle.dump(model, f)

print("모델이 model_rf.pkl 파일로 저장되었습니다.")


# 모델 불러오기
with open("model_rf.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# 불러온 모델로 예측
y_pred = loaded_model.predict(X_test)
print("불러온 모델 예측 완료")


# shap

import shap





import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# 1. 데이터 로딩 및 분할
df = pd.read_csv("./data/train.csv")
X = df.drop(columns=['id', 'passorfail'])
y = df['passorfail']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=1449
)

# 2. 수치형 / 범주형 컬럼 분리
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# 3. 전처리 + 모델 구성
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])
model = make_pipeline(
    preprocessor,
    RandomForestClassifier(n_estimators=100, random_state=42)
)
model.fit(X_train, y_train)

# 4. 전처리 단계와 모델 분리
fitted_preprocessor = model.named_steps["columntransformer"]
rf_model = model.named_steps["randomforestclassifier"]

# 5. 전처리된 데이터 추출
X_transformed = fitted_preprocessor.transform(X_test)
feature_names = fitted_preprocessor.get_feature_names_out()

# 6. SHAP 계산 (TreeExplainer)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_transformed)

# 7. SHAP 요약 시각화
shap.summary_plot(shap_values[1], X_transformed, feature_names=feature_names, plot_type="bar")





## anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score


df = pd.read_csv("./data/train.csv")

# 수치형 변수만 추출
num_data = df.select_dtypes(include=['int64', 'float64']).drop(columns=['passorfail'])

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(num_data)

# 모델 학습
iso = IsolationForest(contamination=0.2, random_state=42)
iso.fit(X_scaled)

# 이상 여부 예측 (-1: 이상치, 1: 정상)
df['anomaly'] = iso.predict(X_scaled)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1: 이상

# 이상치 비율 확인
print(df['anomaly'].value_counts(normalize=True))

# 이상 vs 불량 관계 분석
pd.crosstab(df['passorfail'], df['anomaly'], normalize='index')
f1_score(df['passorfail'],df['anomaly'])
# f1-score 0.51
recall_score(df['passorfail'],df['anomaly'])


################################
# contamination tuning
#################################
contamination_list = np.arange(0.05,0.41,0.05)

# 결과 저장용 리스트
results = []

for contamination in contamination_list:
    # 모델 학습
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X_scaled)
    
    # 이상 여부 예측
    df['anomaly'] = iso.predict(X_scaled)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1: 이상
    
    # 성능 평가
    f1 = f1_score(df['passorfail'], df['anomaly'])
    recall = recall_score(df['passorfail'], df['anomaly'])
    
    # 결과 저장
    results.append({'contamination': contamination, 'f1_score': f1, 'recall_score': recall})

# 결과를 DataFrame으로 보기 좋게 출력
results_df = pd.DataFrame(results)
print(results_df)





# 하이퍼 파라미터 튜닝
# 수치형 변수 추출
num_data = df.select_dtypes(include=['int64', 'float64']).drop(columns=['passorfail'])
y_true = df['passorfail']

# 하이퍼파라미터 후보들
param_grid = {
    'contamination': np.arange(0.05,0.41,0.05),
    'n_estimators': np.arange(80,130,10),
    'max_samples': [0.6, 0.8, 1.0],
    'max_features': [0.6, 0.8, 1.0],
}

# 조합 만들기
from itertools import product
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in product(*values)]

# 결과 저장
results = []

for params in param_combinations:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('iso', IsolationForest(
            contamination=params['contamination'],
            n_estimators=params['n_estimators'],
            max_samples=params['max_samples'],
            max_features=params['max_features'],
            random_state=42
        ))
    ])
    
    pipeline.fit(num_data)
    preds = pipeline.predict(num_data)
    preds = np.where(preds == -1, 1, 0)
    
    recall = recall_score(y_true, preds)
    results.append({
        **params,
        'recall': recall
    })

# 결과 정렬
pd.set_option('display.max_rows', None)

results_df = pd.DataFrame(results).sort_values(by='recall', ascending=False)
print(results_df.head())

results_df.to_csv('isolation_params.csv',index=False)

results_df[results_df['contamination'] == 0.2]



# 최종 모델 저장하는 코드
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, f1_score
import pickle
import numpy as np
df = pd.read_csv("./data/train.csv")

# 수치형 변수 추출
num_data = df.select_dtypes(include=['int64', 'float64']).drop(columns=['passorfail'])
y_true = df['passorfail']

# 파이프라인 구성
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('isolation_forest', IsolationForest(
        contamination=0.2,
        n_estimators=80,
        max_samples=1.0,
        max_features=0.6,
        random_state=42
    ))
])

# 모델 학습
final_pipeline.fit(num_data)

# 예측 수행 (-1: 이상치 → 1로 바꾸기)
y_pred = final_pipeline.predict(num_data)
y_pred = np.where(y_pred == -1, 1, 0)


# 평가 지표 계산
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Recall Score: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 모델 저장
with open("isolation_forest_model.pkl", "wb") as f:
    pickle.dump(final_pipeline, f)

print("모델이 isolation_forest_model.pkl 파일로 저장되었습니다.")



# 모델 불러오기
with open("final_isolation_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# 테스트 데이터에서 수치형 추출
num_test = df.select_dtypes(include=['int64', 'float64'])

# 이상치 예측
df['anomaly'] = model.predict(num_test)
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})


#############


## XGB
import pandas as pd
import shap
import xgboost as xgb
csv_path="data/train.csv"
df = pd.read_csv(csv_path)
drop_cols = [
        'id', 'date', 'time', 'registration_time',
        'line', 'name', 'mold_name', 'upper_mold_temp3', 'lower_mold_temp3'
    ]
X = df.drop(columns=drop_cols + ['passorfail'], errors='ignore')
y = df['passorfail']

for col in X.columns:
    mode_val = X[col].mode(dropna=True)
    X[col] = X[col].fillna(mode_val.iloc[0] if not mode_val.empty else 0)
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category').cat.codes
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)
importances = shap_values.abs.mean(0).values
feature_names = X.columns
sorted_idx = importances.argsort()[::-1]
top_features = feature_names[sorted_idx][:10]


#### GPT
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import shap

# 1. 데이터 불러오기
df = pd.read_csv("data/train.csv")

# 2. 기본 변수 설정
drop_cols = [
    'id', 'date', 'time', 'registration_time',
    'line', 'name', 'mold_name', 'upper_mold_temp3', 'lower_mold_temp3'
]
df = df.drop(columns=drop_cols, errors='ignore')
X = df.drop(columns='passorfail')
y = df['passorfail']

# 3. 컬럼 분류
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# 4. 전처리기 구성
preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='most_frequent'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ]), cat_cols)
])

# 5. 전체 파이프라인 구성
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])



# 6. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 모델 학습
pipeline.fit(X_train, y_train)

# 8. 평가
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))




# 칼럼명 복원 다시 9.
# 1. 전처리된 X_test
preprocessed_X = pipeline.named_steps['preprocessing'].transform(X_test)

# 2. 전처리 후 실제 feature 이름 복원
# 전처리된 feature 이름 얻기
raw_feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()

# 'num__' 또는 'cat__' 제거
feature_names = [name.split("__")[-1] for name in raw_feature_names]



# 3. SHAP explainer 생성
model_only = pipeline.named_steps['model']
explainer = shap.Explainer(model_only, feature_names=feature_names)

# 4. 한 샘플 선택
idx = 0
single_sample = preprocessed_X[idx:idx+1]
shap_values = explainer(single_sample)

# 5. 해석용 DataFrame 생성
shap_df = pd.DataFrame({
    'feature': feature_names,
    'shap_value': shap_values.values[0],
    'value': single_sample[0]
}).sort_values(by='shap_value', key=abs, ascending=False)

print(shap_df.head(10))

shap.plots.bar(shap_values)        # bar plot (기여도 순서)
shap.plots.waterfall(shap_values[0])  # 한 샘플에 대한 상세 흐름












# XGB - model save
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# 1. 데이터 불러오기
df = pd.read_csv("data/train.csv")

# 2. 불필요한 컬럼 제거
drop_cols = [
    'id', 'date', 'time', 'registration_time',
    'line', 'name', 'mold_name', 'upper_mold_temp3', 'lower_mold_temp3'
]
df = df.drop(columns=drop_cols, errors='ignore')

# 3. 피처/타겟 분리
X = df.drop(columns='passorfail')
y = df['passorfail']

# 4. 수치형 / 범주형 컬럼 분리
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# 5. 전처리 파이프라인 정의
preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='most_frequent'), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ]), cat_cols)
])

# 6. 전체 파이프라인 구성
pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# 7. 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 8. 모델 저장
with open("xgb_pipeline_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ 모델이 'xgb_pipeline_model.pkl'로 저장되었습니다.")




df = pd.read_csv('./data/train.csv')

alias = {
        "id": "ID",
        "line": "라인",
        "name": "이름",
        "mold_name": "금형 이름",
        "time": "시간",
        "date": "날짜",
        "count": "생산 수량",
        "working": "작동 여부",
        "emergency_stop": "비상정지",
        "molten_temp": "용탕 온도",
        "facility_operation_cycleTime": "설비 운영 주기",
        "production_cycletime": "생산 주기",
        "low_section_speed": "저속 구간 속도",
        "high_section_speed": "고속 구간 속도",
        "molten_volume": "용탕 체적",
        "cast_pressure": "주조 압력",
        "biscuit_thickness": "비스킷 두께",
        "upper_mold_temp1": "상부 금형 온도1",
        "upper_mold_temp2": "상부 금형 온도2",
        "upper_mold_temp3": "상부 금형 온도3",
        "lower_mold_temp1": "하부 금형 온도1",
        "lower_mold_temp2": "하부 금형 온도2",
        "lower_mold_temp3": "하부 금형 온도3",
        "sleeve_temperature": "슬리브 온도",
        "physical_strength": "물리적 강도",
        "Coolant_temperature": "냉각수 온도",
        "EMS_operation_time": "EMS 작동 시간",
        "registration_time": "등록 시간",
        "passorfail": "불량 여부",
        "tryshot_signal": "시도 신호",
        "mold_code": "몰드 코드",
        "heating_furnace": "가열로"
    }
df = df.rename(columns=alias)
df.to_csv('./data/train_kr.csv', index=False)