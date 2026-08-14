# LG Aimers 9기 Public 910 추출 소스

이 묶음에는 Public Score `910.2060575824`를 기록한 제출 ZIP의 **추출본과
구조 설명만** 포함한다. 후속 연구 코드, 실험 결과, Notebook, 원본 대회
데이터와 Colab 인증 정보는 포함하지 않았다.

## 구성

```text
extracted/
  model/                 # 학습 완료 모델, frozen lookup, 피처 코드
  script.py              # 평가 서버에서 실행되는 실제 추론 코드
  requirements.txt       # 당시 제출물의 원본 의존성 명세
docs/
  ARCHITECTURE_910.md
MANIFEST.sha256
```

## 추론 경로

```text
현재 평가 행
  -> row-local 파생 피처
  -> train-frozen CE lookup
  -> v3_base / hist_current_gap Axis-A 5-class CatBoost 각 3 seed
  -> 두 family 50:50 평균
  -> train-frozen within-season residual 보정 3 seed
  -> train-frozen 팀 이동 D6 보정
  -> control_success 확률
```

평가 행끼리 groupby, rolling, lag, 누적 상태 갱신 또는 test 전체 분포 보정을
하지 않는다. `model/` 안의 일부 파일에는 과거 artifact 생성용 함수도 있지만,
`script.py`의 평가 추론 경로에서는 frozen 파일을 읽어 현재 행에 조회만 한다.

## 구조 확인

상세 설명은 다음 문서를 먼저 읽는다.

```text
docs/ARCHITECTURE_910.md
```

전체 파일의 무결성은 다음 명령으로 확인한다.

```bash
sha256sum -c MANIFEST.sha256
```

## 주의사항

- 이 디렉터리는 **추출된 검토용 소스**이며 제출 ZIP 자체가 아니다.
- 제출하려면 `extracted/`의 내용이 ZIP 최상위가 되도록 별도로 압축해야 한다.
- `feature_manifest.json`의 학습 지표 키를 포함해 `extracted/`는 당시 910
  제출물 그대로이며, 별도의 후속 실험 코드나 결과는 포함하지 않는다.
- 평가 서버의 최신 Python 및 패키지 계약을 제출 전에 다시 확인해야 한다.
- 공식 데이터에서 학습된 모델과 선수별 frozen artifact가 포함되므로 같은 DACON
  등록 팀 안에서만 비공개로 공유하고 공개 저장소에 올리지 않는다.
