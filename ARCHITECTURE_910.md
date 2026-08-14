# Public 910 제출 구조

대상: Public Score `910.2060575824`를 기록한 제출물의 추출본

## 1. 전체 흐름

```text
data/test.csv
  -> engineer_features()
  -> add_ce_features()
  -> v3_base frame
  -> add_history_artifacts()
  -> hist_current_gap frame
  -> 두 CatBoost family의 success 확률
  -> 0.5 * v3_base + 0.5 * hist_current_gap
  -> within-season residual correction
  -> team-transition correction
  -> output/submission.csv
```

모든 학습 모델과 lookup은 평가 전 공식 train에서 생성되어 `model/`에 동결돼
있다. 추론 중 평가 데이터로 모델이나 통계를 다시 fit하지 않는다.

## 2. Strong Head

### Target

원래 이진 성공 여부와 함께 복원한 실패 구조를 사용해 다음 5개 클래스를 학습한다.

```text
0 success
1 reverse_only
2 middle_only
3 reverse_middle
4 other
```

추론 시 `predict_proba` 결과에서 success class 열만 `P(control_success=1)`로
사용한다. success 열 위치는 `feature_manifest.json`의
`success_class_index`를 읽으므로 코드에서 고정 인덱스로 가정하지 않는다.

### Model contract

```text
model              CatBoostClassifier
loss               MultiClass
iterations         96
depth              8
learning_rate      0.055
l2_leaf_reg        7.0
seeds              20260807, 20260808, 20260809
families           v3_base, hist_current_gap
history_weight     0.5
```

family별 3개 모델, 총 6개 분류기가 있다.

```text
p_base = mean(v3_base seed 3개)
p_hist = mean(hist_current_gap seed 3개)
p_current = 0.5 * p_base + 0.5 * p_hist
```

## 3. Row-Local Features

`model/v3_features.py`는 현재 행만으로 다음 문맥을 만든다.

- 볼-스트라이크 count state
- 주자-아웃 base/out state
- 이닝 phase
- 투타 hand matchup
- leverage와 점수차 상태
- cold-start와 history availability
- 최근 1/3/5경기 상태와 장기 상태의 차이
- pitcher/batter success 차이
- pitch-mix entropy와 구종군 비율 차이

이 단계는 다른 평가 행을 읽지 않는다.

## 4. Frozen CE

`model/ce_features.py`는 train에서 생성한 두 lookup을 현재 행에 붙인다.

| Feature | 의미 |
|---|---|
| `ce_p_rate` | pitcher 단위 수축 성공률 |
| `ce_platoon_rate` | pitcher x platoon 수축 성공률 |
| `ce_platoon_dev` | platoon rate와 pitcher rate의 차이 |
| `ce_platoon_logn` | 해당 cell의 log 표본 수 |

사용 파일:

```text
ce_pitcher_entity_2025.csv
ce_platoon_cell_2025.csv
ce_manifest.json
```

미관측 pitcher는 train-frozen global rate를 사용하고, 미관측 cell은 CatBoost가
처리할 수 있도록 일부 값을 NaN으로 유지한다.

## 5. Frozen Historical State

`model/history_artifacts.py`는 `history_artifact_2025.csv`를 현재 행의
`pitcher_id`와 `season`에 many-to-one으로 결합한다.

주요 정보:

- 과거 관측 월·시즌 수
- 마지막 관측 시즌과 현재 시즌의 gap
- control/pitch-mix 상태의 last, mean, trend, volatility
- 현재 행의 공식 asof 값과 frozen preseason 값의 차이인 `histgap_*`

artifact는 평가 전에 만들어져 있으므로 추론 시 test의 다른 행을 과거 기록으로
사용하지 않는다.

## 6. Within-Season Residual Correction

Strong head의 `p_current`에 CatBoostRegressor 3개의 평균 보정량을 더한다.

```text
correction_seed = clip(model.predict(ws_features), -cap, +cap)
prediction_seed = clip(p_current + correction_seed, 0, 1)
prediction = mean(prediction_seed 3개)
```

`ws_features`는 현재 행의 공식 누적 asof 상태와
`within_season_preseason_2025.csv`의 frozen preseason 상태 차이로 계산한다.

사용 파일:

```text
within_season_full_seed20260807.cbm
within_season_full_seed20260808.cbm
within_season_full_seed20260809.cbm
within_season_preseason_2025.csv
within_season_manifest.json
```

`within_season_features.py`에는 과거 artifact 생성 및 학습 함수도 보존되어 있지만,
평가 서버의 `script.py`는 `add_within_season_features()`와 frozen 모델 추론만
호출한다.

## 7. Team-Transition Correction

현재 행의 `pitcher_team_id`와 train-frozen 이전 팀 lookup을 비교한다.

```text
team_changed = previous_team exists and previous_team != current_row_team
final = clip(prediction + team_changed * frozen_correction, 0, 1)
```

사용 파일:

```text
team_transition_preseason_2025.csv
team_transition_manifest.json
```

이 역시 현재 행과 frozen lookup만 사용한다.

## 8. Output

`script.py`는 test의 `row_id`와 계산된 확률을 연결한 뒤
`sample_submission.csv`의 순서에 맞춰 다음 파일을 만든다.

```text
output/submission.csv
```

저장 전 다음을 검사한다.

- test와 prediction 행 수 일치
- 모든 확률이 finite
- 모든 확률이 `[0, 1]`
- sample submission의 모든 `row_id`가 매핑됨

## 9. 평가 행 독립성

최종 확률은 아래 정보만 사용한다.

- 현재 평가 행의 공식 입력
- 현재 행에서 만든 deterministic feature
- 공식 train에서 미리 만든 frozen model과 lookup

추론 경로는 다른 평가 행을 이용한 groupby, rolling, lag, cumulative update,
빈도, rank, quantile, calibration을 사용하지 않는다. 같은 행을 단독으로 넣거나
다른 행과 함께 넣어도 확률이 같아야 한다.

## 10. 파일 역할

```text
extracted/
  script.py
  requirements.txt
  model/
    *.cbm                              학습 완료 CatBoost 모델
    feature_manifest.json              strong head 계약
    v3_features.py                     row-local 파생 피처
    history_artifacts.py               frozen history lookup 결합
    history_artifact_2025.csv           frozen history artifact
    ce_features.py                     frozen CE lookup 결합
    ce_*                               CE artifact와 manifest
    within_season_features.py          current-vs-preseason 피처
    within_season_*                    residual 모델/artifact/manifest
    team_transition_*                  팀 이동 artifact/manifest
```
