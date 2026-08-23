# Ultimate Model (Leakage-Free)

## 1. Description (개요)
미래 정보 참조(Data Leakage)를 원천 차단하고 순수 야구 피처만을 사용하였으며, 50-Trial Optuna 최적화를 통해 스태킹 과적합을 해결한 완성형 버전입니다.

## 2. Architecture (모델 구조)
**Level 1 Base**:
- 누수 없는 독립적인 베이스 모델

**Level 2 이종 앙상블**:
- CatBoost, LightGBM, XGBoost 3개의 모델을 동일한 순수 피처셋으로 학습

**Level 3 Stacking**:
- 3개 모델의 OOF 확률을 취합하여 메타 모델 학습

## 3. Features Used (사용된 피처 엔지니어링)
- 순수 야구 피처: Leakage 위험이 있는 메타/통계 피처를 완전히 배제하고 트랙맨 데이터, 이닝, 점수차, 베이스 상황 등 순수 야구 상황 정보만으로 구성

## 4. Hyperparameter Tuning (튜닝 세부사항)
- **50-Trial Optuna TPE(Tree-structured Parzen Estimator) 최적화**: 하이퍼파라미터 탐색 공간을 정의하고, 50번의 Trial을 통해 베이지안 최적화 기법으로 `learning_rate`, `depth`, `l2_leaf_reg`, `min_child_samples` 등 핵심 파라미터를 자동 탐색 및 최적화
- **OOF 예측 (Out-of-Fold)**: 과적합을 막기 위해 철저히 K-Fold 기반의 OOF 확률만을 모아서 Level 3에 전달
