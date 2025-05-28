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