import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 데이터 로드 및 df 변수에 할당
df = pd.read_csv('c:/Users/KimHome/Downloads/heart_raw data.csv')

# 결측치 처리
df['sex'] = df['sex'].fillna('Male')
df['fbs'] = df['fbs'].fillna('FALSE')
df['exang'] = df['exang'].fillna('FALSE')

# 문자열을 숫자형으로 변환
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1}).astype(int)
df['fbs'] = pd.to_numeric(df['fbs'].replace({'TRUE': 1, 'FALSE': 0}), errors='coerce').fillna(0).astype(int)
df['exang'] = pd.to_numeric(df['exang'].replace({'TRUE': 1, 'FALSE': 0}), errors='coerce').fillna(0).astype(int)

# 범주형 데이터를 더미 변수로 변환
df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'thal'], drop_first=True)

# 수치형 변수의 결측치를 평균값으로 채우기
df.fillna(df.mean(), inplace=True)

# 종속 변수 및 독립 변수 선택
y = df['num']  # 'num' 열을 종속 변수로 가정
X = df.drop('num', axis=1)  # 'num' 열을 제외한 모든 열을 독립 변수로 설정

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1', C=1.0)  # 추가 파라미터 설정
model.fit(X_train, y_train)

# 모델 평가
predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# 시각화
# 종속 변수의 분포
plt.figure(figsize=(10, 6))
sns.countplot(x='num', data=df)
plt.title('Distribution of Target Variable (num)')
plt.show()

# 성별 분포
plt.figure(figsize=(10, 6))
sns.countplot(x='sex', data=df)
plt.title('Distribution of Sex')
plt.show()

# 나이 분포
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# 상관 행렬
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 예측 분포
plt.figure(figsize=(10, 6))
sns.countplot(x=predictions)
plt.title('Distribution of Predictions')
plt.show()

# 실제 값 분포
plt.figure(figsize=(10, 6))
sns.countplot(x=y_test)
plt.title('Distribution of Actual Values')
plt.show()

