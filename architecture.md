# 🏆 최종 모델 아키텍처: 3중 앙상블 앵커 + 2단 투수 잔차 보정기

본 문서는 데이콘 제구력 예측(control_success) 과제를 위해 구축된 최종 머신러닝 파이프라인의 아키텍처 구조도입니다.

## 1. 아키텍처 다이어그램 (Mermaid)

```mermaid
graph TD
    %% 스타일 정의
    classDef dataNode fill:#f8f9fa,stroke:#dee2e6,stroke-width:2px,rx:10px,ry:10px;
    classDef prepNode fill:#e3f2fd,stroke:#90caf9,stroke-width:2px;
    classDef modelNode fill:#e8f5e9,stroke:#81c784,stroke-width:2px,rx:5px,ry:5px;
    classDef calibNode fill:#fff3e0,stroke:#ffb74d,stroke-width:2px,stroke-dasharray: 5 5;
    classDef blendNode fill:#f3e5f5,stroke:#ce93d8,stroke-width:2px;
    classDef resNode fill:#ffebee,stroke:#ef9a9a,stroke-width:2px,rx:10px,ry:10px;
    classDef finalNode fill:#212121,stroke:#000000,stroke-width:3px,color:#ffffff,rx:10px,ry:10px;

    %% 노드 정의
    A["투구 데이터 (2025 Test)"]:::dataNode
    
    subgraph Preprocessing [데이터 전처리 및 임베딩]
        P1["파생 변수 생성 (Rolling Stats, 폼 변화량 등)"]:::prepNode
        P2["Ordinal Encoding (범주형 11종)"]:::prepNode
        P3["Robust Scaling (연속형)"]:::prepNode
    end
    
    subgraph Level1 [1단계: 19~24 앵커 앙상블 (24년 1.5배 가중)]
        CB["CatBoost (Tree)"]:::modelNode
        MLP["RealMLP (nn.Embedding + Dense)"]:::modelNode
        FTT["FT-Transformer (Tokenization + Attention)"]:::modelNode
        
        ISO_MLP["Isotonic Calibration"]:::calibNode
        ISO_FTT["Isotonic Calibration"]:::calibNode
        
        W["최적 가중 평균 앙상블"]:::blendNode
    end

    subgraph Level2 [2단계: 투수 고유 잔차 보정기]
        RES_FEAT["투수 잔차 이력 6종 추출 (hist_rows, mean, slope 등)"]:::resNode
        CB_RES["c_b (2024년 전체 잔차 타깃)"]:::resNode
        CV_RES["c_v (2024년 7월 이후 잔차 타깃)"]:::resNode
    end

    FINAL["최종 제구 확률 (Base + 1.2*c_b - 0.3*c_v)"]:::finalNode

    %% 연결 (데이터 흐름)
    A --> P1
    P1 --> P2
    P1 --> P3
    
    P1 --> CB
    P2 --> MLP
    P3 --> MLP
    P2 --> FTT
    P3 --> FTT

    CB --> W
    MLP --> ISO_MLP
    FTT --> ISO_FTT
    
    ISO_MLP --> W
    ISO_FTT --> W
    
    W --> FINAL
    
    A --> RES_FEAT
    RES_FEAT --> CB_RES
    RES_FEAT --> CV_RES
    
    CB_RES -.->|"+1.2 곱하기"| FINAL
    CV_RES -.->|"-0.3 곱하기"| FINAL
```

---

## 2. 텍스트 구조도 (ASCII Art)

```text
[ ⚾ 투구 데이터 (2025 Test) ]
             │
             ▼
=========================================
[ 🛠️ 전처리 및 임베딩 ]
 ├─ 파생 변수 (Rolling Stats 등)
 ├─ Ordinal Encoding (범주형 11종)
 └─ Robust Scaling (연속형 숫자)
=========================================
             │
             ▼
=========================================
[ 🚀 1단계: 앵커 앙상블 (19~24년 데이터 학습) ]
 ├─ 🌳 CatBoost (트리 모델)
 ├─ 🧠 RealMLP (신경망) ──▶ [확률보정: Isotonic]
 └─ 🤖 FTT (트랜스포머) ──▶ [확률보정: Isotonic]
             │
             ▼
      [ ⚖️ 최적 가중 평균 (w0, w1, w2) ]
             │
             ▼ (Base 예측값)
=========================================
             │
             ▼
=========================================
[ 🎯 2단계: 투수 고유 잔차 보정기 ]
 ├─ 📊 투수 잔차 이력 (hist_rows, residual_mean 등 6종)
 │     (※ 구종, 볼카운트 등 다른 변수는 철저히 배제)
 │
 ├─ 🟢 c_b (2024년 1년치 잔차 타깃)
 └─ 🔵 c_v (2024년 7월 이후 잔차 타깃)
=========================================
             │
             ▼
[ 🏁 최종 제구 확률 계산 (상수 α=1.2, β=-0.3) ]
 Base 예측값 + (1.2 × c_b) - (0.3 × c_v)
             │
             ▼
     [ 🏆 최종 제출물 (submission.csv) ]
```

## 3. 핵심 설계 의도 요약
1. **딥러닝 임베딩 적용**: 트리 모델(CatBoost)이 놓치는 복잡한 비선형적 상호작용을 RealMLP와 FT-Transformer의 임베딩 레이어 및 멀티헤드 어텐션으로 추출.
2. **신경망 자만심(Overconfidence) 억제**: Brier Score 극대화를 위해 딥러닝 예측값에 Isotonic Regression을 적용하여 실제 경험적 빈도수(Empirical Probability)와 완벽히 동기화.
3. **이중 적용(Double-counting) 방어**: 2단계 잔차 보정기는 구종이나 타자 상황 등 일반 변수를 철저히 배제하고, 오직 '투수별 잔차 이력(6종)'만 학습하여 분산(Variance)이 폭발하는 것을 완벽히 방지함.
