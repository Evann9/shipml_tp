# 선박 유형 분류

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드
# ais_data_ultra_cleaned.csv를 참고하여 데이터 로드
df = pd.read_csv("ais_data_ultra_cleaned.csv")

# 2. 특성 및 타깃 설정
# sog, cog, heading, width, length, draught를 독립변수로 사용
features = ["sog", "cog", "heading", "width", "length", "draught"]
target = "shiptype"

# 결측치 제거 (분류 모델 학습을 위해)
df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

# 3. 데이터 분할 (학습용/테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 모델 생성 및 학습 (Random Forest 분류기)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 예측 및 평가
y_pred = model.predict(X_test)

print("--------- Model Evaluation ---------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}") # Accuracy: 0.9931
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Classification Report:
#                  precision    recall  f1-score   support

#           Cargo       0.99      1.00      0.99     34166
#        Dredging       0.99      1.00      1.00       794
#         Fishing       1.00      1.00      1.00      2278
#             HSC       1.00      0.99      1.00       738
# Law enforcement       1.00      1.00      1.00       302
#        Military       0.99      1.00      1.00       839
#       Passenger       1.00      1.00      1.00      1928
#           Pilot       1.00      1.00      1.00       395
#        Pleasure       1.00      1.00      1.00        15
#             SAR       1.00      0.92      0.96        93
#         Sailing       1.00      0.81      0.89        31
#          Tanker       0.99      0.98      0.99     14836
#          Towing       0.94      0.93      0.93       161
#             Tug       0.99      1.00      0.99      1529

#        accuracy                           0.99     58105
#       macro avg       0.99      0.97      0.98     58105
#    weighted avg       0.99      0.99      0.99     58105

# 6. 특성 중요도 시각화
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importances for Ship Type Classification")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# 7. 혼동 행렬 (Confusion Matrix) 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 8. 새로운 데이터 예측 예시
# sog, cog, heading, width, length, draught 순서
new_data = pd.DataFrame([
    [12.5, 180.0, 175.0, 32.0, 250.0, 11.5], # Cargo 예상
    [0.1, 45.0, 90.0, 10.0, 30.0, 3.5],      # Tug/Fishing 예상
    [25.0, 270.0, 270.0, 25.0, 180.0, 7.0]   # Passenger 예상
], columns=features)

new_preds = model.predict(new_data)
print("\n--------- New Data Predictions ---------")
for i, pred in enumerate(new_preds):
    print(f"Data {i+1} Predicted Type: {pred}")

# 9. 모델 비교 및 성능 개선 (추가)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# 스케일링 (Logistic Regression 등 선형 모델용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--------- Model Comparison ---------")
# Baseline: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
lr_model.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr_model.predict(X_test_scaled))
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
# Logistic Regression Accuracy: 0.7301

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42) # 속도를 위해 n_estimators 조절
gb_model.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb_model.predict(X_test))
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
# Gradient Boosting Accuracy: 0.8819

# 결과 비교 요약
comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression', 'Gradient Boosting'],
    'Accuracy': [accuracy_score(y_test, y_pred), lr_acc, gb_acc]
})
print("\n", comparison)
#             Model       Accuracy
# 0        Random Forest  0.993099
# 1  Logistic Regression  0.730144
# 2    Gradient Boosting  0.881921


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

# XGBoost와 LightGBM은 타깃 레이블이 숫자인 것을 선호함
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# XGBoost 모델
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train_encoded)
xgb_acc = accuracy_score(y_test_encoded, xgb_model.predict(X_test))
print(f"XGBoost Accuracy: {xgb_acc:.4f}")
# XGBoost: 0.9775

# LightGBM 모델
lgbm_model = LGBMClassifier(n_estimators=100, random_state=42)
lgbm_model.fit(X_train, y_train)
lgbm_acc = accuracy_score(y_test, lgbm_model.predict(X_test))
print(f"LightGBM Accuracy: {lgbm_acc:.4f}")

# 모델별 정확도 순위:
# 1. Random Forest (0.9931): 데이터의 비선형 관계와 이상치에 강하며, 현재 데이터셋의 특성(물리적 크기 차이 등)을 가장 잘 포착함.
# 2. XGBoost (0.9775): 강력한 성능을 보이나 하이퍼파라미터 튜닝 없이 RF의 기본 성능을 넘지 못함.
# 3. Gradient Boosting (0.8819): 순차적 학습 모델이나, 기본 설정(n_estimators=50)이 RF에 비해 충분하지 않았을 가능성이 큼.
# 4. Logistic Regression (0.7301): 선형 모델로서, 선박 종류 간의 복잡한 비선형 경계를 구분하는 데 한계가 있음.
# 5. LightGBM (0.6488): 대용량 데이터에 최적화되어 있으나, 현재 데이터의 불균형(Cargo/Tanker 비중 매우 높음)이나 
#    리프 중심(Leaf-wise) 성장 방식이 소수 클래스(Sailing, SAR 등) 예측에서 오차를 발생시켰을 가능성이 큼.

