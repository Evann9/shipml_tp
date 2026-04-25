# AIS Ship Type and Route Prediction System

AIS 데이터를 활용해 선박 종류를 분류하고, 예측 항로와 미래 좌표를 Flask 웹 지도에서 확인하는 프로젝트입니다.

## 핵심 파일

- `shipml/type_anal/ais_ship_type_with_mmsi.csv`: MMSI 기준 Group Split 평가용 선종 데이터
- `shipml/type_anal/ais_ship_type_features.csv`: 기존 row-level 선종 분류 실험 데이터
- `shipml/type_anal/train_ship_type_classifier_group_split.py`: MMSI Group Split 기반 선종 모델 평가 및 저장
- `shipml/type_anal/tune_ship_type_classifier_group_split.py`: 선종 모델 하이퍼파라미터 튜닝
- `shipml/type_anal/add_ship_type_predictions_to_routes.py`: 항로 예측 결과에 선종 예측 추가
- `shipml/type_anal/export_ship_type_classifier_reports.py`: feature importance, confusion matrix 산출
- `shipml/route_anal/train_future_position_regressor.py`: 1/2/3시간 후 미래 좌표 예측 모델 학습
- `shipml/web/app.py`: Flask + Leaflet 웹 지도

## 주요 산출물

- `shipml/type_anal/outputs/ship_type_classifier_group_split_metrics.json`
- `shipml/type_anal/outputs/ship_type_classifier_class_metrics.csv`
- `shipml/type_anal/outputs/ship_type_classifier_confusion_pairs.csv`
- `shipml/type_anal/outputs/ship_type_classifier_feature_importance.png`
- `shipml/type_anal/outputs/ship_type_classifier_confusion_matrix.png`
- `shipml/route_anal/outputs/route_predictions_with_types.csv`
- `shipml/route_anal/outputs/future_position_forecast.csv`

## 대용량 모델 파일

`.joblib` 모델 파일은 Git에 올리지 않고 GitHub Release에 보관합니다.

Release tag:

```text
shipml-model-artifacts-v1
```

업로드:

```powershell
$env:GITHUB_TOKEN="ghp_..."
.\scripts\upload_model_artifacts_to_github_release.ps1
```

다운로드:

```powershell
.\scripts\download_model_artifacts_from_github_release.ps1
```

## 실행 순서

```powershell
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\type_anal\train_ship_type_classifier_group_split.py --models random_forest xgboost --compare-random-split
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\type_anal\add_ship_type_predictions_to_routes.py --model shipml\type_anal\outputs\ship_type_classifier_group_split.joblib --metrics shipml\type_anal\outputs\ship_type_classifier_group_split_metrics.json
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\type_anal\export_ship_type_classifier_reports.py
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\route_anal\train_future_position_regressor.py
C:\Users\green\anaconda3\envs\myproject\python.exe -m flask --app shipml.web.app:app run --host 127.0.0.1 --port 5000 --no-reload
```

웹 접속:

```text
http://127.0.0.1:5000
```

## 현재 기준 성능

- 선종 분류 모델: RandomForest, MMSI Group Split 기준 Accuracy 0.8453, Macro F1 0.7378
- 미래 좌표 모델: RandomForestRegressor, MMSI GroupShuffleSplit 기준 평균 오차 1h 2.80km / 2h 3.88km / 3h 5.24km

## 해석 포인트

- 기존 row-level split의 높은 정확도는 같은 MMSI가 train/test에 섞인 데이터 누수 가능성이 있습니다.
- 웹의 `예측 확신도`는 개별 예측에 대한 모델 확률이며, 전체 정확도는 Accuracy/Macro F1로 따로 봐야 합니다.
- 예측 항로 패턴 선은 평균 중심선 대신 대표 실제 AIS 항적을 우선 표시해 육지 관통 문제를 줄였습니다.
