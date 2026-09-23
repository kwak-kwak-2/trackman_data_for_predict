# RC3 P2 1072.563 코드 전용 공유본

- 기준 Public score: `1072.5630000262`
- 기준 모델: RC3 P2
- 용도: 동일 팀 내부의 모델 구조 및 구현 검토

이 공유본에는 Python 소스와 `requirements.txt`만 있다. 다음 항목은 포함하지 않았다.

- 모델 가중치와 학습 데이터
- 선수별 동결 통계와 JSON manifest
- OOF 예측 및 실험 결과
- 개인 로컬 경로와 Google Drive 경로
- 파일 SHA 검증
- 패키징, 제출물 조립, 감사 및 스모크 테스트 코드
- Colab 런처와 과거 run 디렉터리 정보

## 구조

```text
inference/
  script.py
  model/*.py

training/
  a4_anchor_reference.py
  rc3_corrector_reference.py
  residual_core/
```

`inference/script.py`와 `inference/model/*.py`는 실제 최고점 제출의 추론 및 피처
로직이다. 실행에 필요한 모델·통계·manifest는 의도적으로 제외했으므로 이 폴더만으로
기존 제출을 실행할 수는 없다.

`training/*_reference.py`는 개인 경로와 실행 인프라를 제거한 학습 핵심이다.
`training/residual_core/`에는 잔차 이력 집계, CatBoost/신경망 모델 및 RC3 variant
학습 로직을 남겼다.

최종 결합식은 다음과 같다.

```python
final = clip(a4_anchor + 1.2 * base_corrector - 0.3 * recent_h2_corrector, 0, 1)
```

추론은 평가 대상 행 자신의 값과 학습 데이터에서 미리 만든 동결 artifact만 사용해야
한다. 평가 데이터의 다른 행을 이용한 집계, rolling, 보정은 허용되지 않는다.
