"""
원본 synthetic 데이터에 의학적 근거 기반 조건부 확률을 주입해
현실적인 상관관계를 가진 데이터셋을 생성합니다.

원본 파일: data/drug_side_effects_100k_dataset.csv
출력 파일: data/drug_side_effects_augmented.csv
"""

import pandas as pd
import numpy as np

SEED = 42
rng  = np.random.default_rng(SEED)

INPUT  = 'data/drug_side_effects_100k_dataset.csv'
OUTPUT = 'data/drug_side_effects_augmented.csv'

# ── 의학적 근거 기반 규칙 ────────────────────────────────────
# 흡연율: 심장질환·고혈압은 높고, 천식은 낮음
SMOKER_PROB = {
    'Heart Disease': 0.65,
    'Hypertension':  0.55,
    'Diabetes':      0.35,
    'Kidney Disease':0.30,
    'Asthma':        0.15,
}

# 음주율: 심장질환·고혈압은 높고, 당뇨·신장질환은 낮음
ALCOHOL_PROB = {
    'Heart Disease': 0.60,
    'Hypertension':  0.50,
    'Asthma':        0.40,
    'Diabetes':      0.25,
    'Kidney Disease':0.20,
}

# 중증도 확률 [Mild, Moderate, Severe]
# 심장질환·신장질환은 Severe 비율 높음
SEVERITY_PROB = {
    'Heart Disease': [0.20, 0.35, 0.45],
    'Kidney Disease':[0.25, 0.35, 0.40],
    'Hypertension':  [0.30, 0.40, 0.30],
    'Diabetes':      [0.40, 0.35, 0.25],
    'Asthma':        [0.55, 0.30, 0.15],
}

# 나이가 많을수록 Severe 확률 증가 (60세 이상 가중치 추가)
AGE_SEVERE_BOOST = 0.15   # 60세 이상이면 Severe 확률 +15%p

# 흡연 + 음주 동시인 경우 Severe 추가 부스트
LIFESTYLE_BOOST  = 0.10

# 약물별 Severe 위험도 조정 (상대적 위험)
DRUG_SEVERE_BOOST = {
    'Metformin':     0.00,
    'Paracetamol':  -0.05,
    'Ibuprofen':     0.05,
    'Amoxicillin':   0.00,
    'Atorvastatin':  0.05,
    'Omeprazole':   -0.05,
    'Lisinopril':    0.10,
    'Amlodipine':    0.10,
    'Insulin':       0.15,
    'Sertraline':    0.10,
}


def assign_smoker(row):
    prob = SMOKER_PROB.get(row['chronic_condition'], 0.40)
    return 'Yes' if rng.random() < prob else 'No'


def assign_alcohol(row):
    prob = ALCOHOL_PROB.get(row['chronic_condition'], 0.40)
    return 'Yes' if rng.random() < prob else 'No'


def assign_severity(row):
    base = SEVERITY_PROB.get(row['chronic_condition'], [0.40, 0.35, 0.25]).copy()

    # 나이 보정
    if row['age'] >= 60:
        base[0] -= AGE_SEVERE_BOOST
        base[2] += AGE_SEVERE_BOOST

    # 생활습관 보정
    if row['smoker'] == 'Yes' and row['alcohol_use'] == 'Yes':
        base[0] -= LIFESTYLE_BOOST
        base[2] += LIFESTYLE_BOOST

    # 약물 보정
    boost = DRUG_SEVERE_BOOST.get(row['drug_name'], 0.0)
    base[0] -= boost
    base[2] += boost

    # 음수 방지 후 정규화
    base = [max(0.0, p) for p in base]
    total = sum(base)
    base  = [p / total for p in base]

    return rng.choice(['Mild', 'Moderate', 'Severe'], p=base)


# ── 실행 ─────────────────────────────────────────────────────
print("원본 데이터 로드 중...")
df = pd.read_csv(INPUT)
print(f"  {df.shape[0]}행 × {df.shape[1]}열")

print("\n원본 중증도 분포:")
print(df['severity'].value_counts())

print("\n상관관계 주입 중...")
df['smoker']      = df.apply(assign_smoker,   axis=1)
df['alcohol_use'] = df.apply(assign_alcohol,  axis=1)
df['severity']    = df.apply(assign_severity, axis=1)

print("\n증강 후 중증도 분포:")
print(df['severity'].value_counts())

print("\n증강 후 기저질환별 흡연율:")
print((df.groupby('chronic_condition')['smoker']
         .apply(lambda x: (x == 'Yes').mean() * 100)
         .round(1)
         .to_string()))

df.to_csv(OUTPUT, index=False)
print(f"\n저장 완료: {OUTPUT}")
