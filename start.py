import pandas as pd

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
sub_df = pd.read_csv('./data/submit.csv')

train_df.info()



train_df.select_dtypes('number')

import matplotlib.pyplot as plt
import seaborn as sns
num_cols = train_df.select_dtypes(include='number').columns

# 반복문으로 히스토그램 출력
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_df[col].dropna(), bins=30, kde=False)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 예: 타겟 y가 0/1인 이진 변수
import numpy as np
corr = train_df.corr(numeric_only=True)
corr_with_target = np.abs(corr['passorfail']).sort_values(ascending=False)
print(corr_with_target)

test_df.info()

train_df['passorfail'].value_counts(normalize=True)
train_df.head()


for col in num_cols:
    if col == 'passorfail':
        continue  # 타겟 변수는 제외
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_df, x='passorfail', y=col)
    plt.title(f"{col} vs passorfail (Boxplot)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for col in num_cols:
    if col == 'passorfail':
        continue
    plt.figure(figsize=(6, 4))
    sns.violinplot(data=train_df, x='passorfail', y=col)
    plt.title(f"{col} vs passorfail (Violinplot)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
import matplotlib.pyplot as plt
import seaborn as sns

# 수치형 변수 목록에서 'passorfail' 제외
num_cols = train_df.select_dtypes(include='number').columns.drop('passorfail')

# subplot 구성 정보
n_cols = 3  # 한 행에 3개씩
n_rows = -(-len(num_cols) // n_cols)  # 전체 변수를 행 단위로 나누기 (올림 나눗셈)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()  # 2D → 1D 배열로 펼치기

for i, col in enumerate(num_cols):
    sns.boxplot(data=train_df, x='passorfail', y=col, ax=axes[i])
    axes[i].set_title(f"{col} vs passorfail")
    axes[i].grid(True)

# 남는 subplot 숨기기 (변수보다 subplot이 더 많은 경우)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()




# 날짜 전처리

train_df['date'] = pd.to_datetime(train_df['time'] + " "+ train_df['date'])





### 날짜 관련


train_df.info()
train_df = train_df.drop(['time'],axis=1)

# 범주형 변수
# 범주형 변수만 선택
cat_cols = train_df.select_dtypes(include='object').columns

# subplot 구성
n_cols = 2
n_rows = -(-len(cat_cols) // n_cols)  # 올림 나눗셈
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    # 비율 기반 막대그래프 (범주 vs 타겟 평균)
    sns.barplot(data=train_df, x=col, y='passorfail', ax=axes[i])
    axes[i].set_title(f"{col} vs passorfail (mean by category)")
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].set_ylabel("Probability of pass (mean)")
    axes[i].set_ylim(0, 1)

# 남는 subplot 삭제
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()






train_df.groupby(['heating_furnace'])['passorfail'].mean()
# B가 조금 더 높다. 


train_df.groupby(['working'])['passorfail'].mean()
# 정지일 때 불량 아주 높음.

train_df.groupby(['working'])['passorfail'].mean()
train_df.info()



def group_target_mean(df, target_col='passorfail'):
    cat_cols = df.select_dtypes(include='object').columns
    result_dict = {}

    for col in cat_cols:
        group = df.groupby(col)[target_col]
        
        # 평균 (pass 비율), 개수
        group_mean = group.mean()
        group_count = group.count()
        
        # 비율 계산
        group_ratio = group_count / len(df)

        # 병합
        result = pd.concat([group_mean, group_count, group_ratio], axis=1)
        result.columns = [f'{target_col}_mean', 'count', 'ratio']
        result = result.sort_values(by=f'{target_col}_mean', ascending=False)

        print(f"\n[ {col} vs {target_col} ]")
        print(result)

        result_dict[col] = result.reset_index()

    return result_dict



train_df.select_dtypes('object').columns
results = group_target_mean(train_df)

results['time'].sort_values('ratio', ascending=False)  # 특정 결과 확인

results['date'].sort_values('ratio', ascending=False)  # 특정 결과 확인

(train_df[(train_df['molten_volume'] == 49)]['passorfail']).sum()

train_df[train_df['upper_mold_temp3'] > 1400]['passorfail'].mean()
train_df['passorfail'].mean()

train_df[train_df['low_section_speed'] > 60000]['passorfail']
train_df[train_df['lower_mold_temp3'] > 60000]['passorfail']
train_df[train_df['physical_strength'] > 60000]['passorfail']

sns.boxplot(train_df[train_df['low_section_speed'] < 60000]['low_section_speed'])
train_df['low_section_speed'].describe()

train_df['molten_temp'].describe()


train_df['low_section_speed'].isna().sum()
train_df['lower_mold_temp3'].isna().sum()
train_df['physical_strength'].isna().sum()
train_df[train_df['lower_mold_temp3'].isna()]['passorfail'].sum()

'lower_mold_temp3'
'physical_strength'


train_df.groupby(['passorfail'])['count'].mean()
train_df.groupby(['passorfail'])['id'].mean()



