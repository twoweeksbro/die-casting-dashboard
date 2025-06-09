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
num_test = test_df.select_dtypes(include=['int64', 'float64'])

# 이상치 예측
test_df['anomaly'] = model.predict(num_test)
test_df['anomaly'] = test_df['anomaly'].map({1: 0, -1: 1})
