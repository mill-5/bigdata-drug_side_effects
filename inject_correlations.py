"""
원본 synthetic 데이터에 의학적 근거 기반 조건부 확률을 주입해
현실적인 상관관계를 가진 데이터셋을 생성합니다.

원본 파일: data/drug_side_effects_100k_dataset.csv
출력 파일: data/drug_side_effects_augmented.csv
"""

import pandas as pd
import numpy as np

# 재현 가능한 결과를 위해 랜덤 시드 고정 (42는 관례적으로 많이 쓰는 값)
SEED = 42
rng  = np.random.default_rng(SEED)

# 입력/출력 파일 경로
INPUT  = 'data/drug_side_effects_100k_dataset.csv'
OUTPUT = 'data/drug_side_effects_augmented.csv'

# ── 의학적 근거 기반 규칙 ────────────────────────────────────

# 기저질환별 흡연 확률 설정
# 심장질환·고혈압 환자는 흡연율이 높고, 천식 환자는 흡연하면 악화되므로 낮게 설정
SMOKER_PROB = {
    'Heart Disease': 0.65,   # 심장질환 환자의 65%가 흡연자
    'Hypertension':  0.55,   # 고혈압 환자의 55%가 흡연자
    'Diabetes':      0.35,
    'Kidney Disease':0.30,
    'Asthma':        0.15,   # 천식은 흡연이 직접 악화 요인이라 낮음
}

# 기저질환별 음주 확률 설정
# 당뇨·신장질환 환자는 음주가 매우 위험하므로 낮게 설정
ALCOHOL_PROB = {
    'Heart Disease': 0.60,
    'Hypertension':  0.50,
    'Asthma':        0.40,
    'Diabetes':      0.25,   # 혈당 문제로 음주율 낮음
    'Kidney Disease':0.20,   # 신장 부담으로 음주율 매우 낮음
}

# 기저질환별 중증도 확률 [Mild 비율, Moderate 비율, Severe 비율]
# 세 값의 합이 1.0이 되어야 함
# 심장질환·신장질환은 Severe 비율 높음
SEVERITY_PROB = {
    'Heart Disease': [0.10, 0.25, 0.65],  # 65%가 중증
    'Kidney Disease':[0.15, 0.30, 0.55],  # 55%가 중증
    'Hypertension':  [0.25, 0.40, 0.35],
    'Diabetes':      [0.35, 0.40, 0.25],
    'Asthma':        [0.50, 0.35, 0.15],  # 15%만 중증
}

# 나이가 많을수록 중증 위험 증가: 60세 이상이면 Severe 확률 +20%p
AGE_SEVERE_BOOST = 0.20

# 흡연과 음주를 동시에 하면 Severe 확률 추가 +15%p
# 두 가지 나쁜 생활습관이 겹치면 위험이 더 커짐
LIFESTYLE_BOOST  = 0.15

# 약물별 Severe 위험도 조정값 (양수=위험, 음수=안전)
DRUG_SEVERE_BOOST = {
    'Metformin':     0.00,   # 당뇨 1차 치료제, 비교적 안전
    'Paracetamol':  -0.05,   # 타이레놀, 안전한 편
    'Ibuprofen':     0.05,   # 소염진통제, 위장·신장 부담 있음
    'Amoxicillin':   0.00,   # 항생제, 중립
    'Atorvastatin':  0.05,   # 고지혈증약, 간독성 가능
    'Omeprazole':   -0.05,   # 위산억제제, 안전한 편
    'Lisinopril':    0.10,   # 고혈압약, 신장 기능 영향
    'Amlodipine':    0.10,   # 칼슘채널차단제, 부작용 주의
    'Insulin':       0.20,   # 저혈당 위험으로 Severe 가장 높음
    'Sertraline':    0.10,   # 항우울제, 상호작용 주의
}


def assign_smoker(row):
    # 해당 기저질환의 흡연 확률 가져오기 (없는 기저질환이면 기본값 40%)
    prob = SMOKER_PROB.get(row['chronic_condition'], 0.40)
    # prob 확률로 Yes, 나머지 확률로 No 반환
    return 'Yes' if rng.random() < prob else 'No'


def assign_alcohol(row):
    # 해당 기저질환의 음주 확률 가져오기 (없는 기저질환이면 기본값 40%)
    prob = ALCOHOL_PROB.get(row['chronic_condition'], 0.40)
    return 'Yes' if rng.random() < prob else 'No'


def assign_severity(row):
    # 1단계: 기저질환에 따른 기본 중증도 확률 가져오기
    # .copy()로 원본 리스트를 복사해서 수정해도 원본이 안 바뀌게 함
    base = SEVERITY_PROB.get(row['chronic_condition'], [0.40, 0.35, 0.25]).copy()
    # base[0] = Mild 확률, base[1] = Moderate 확률, base[2] = Severe 확률

    # 2단계: 나이 보정 — 60세 이상이면 Severe 확률 +20%, Mild 확률 -20%
    if row['age'] >= 60:
        base[0] -= AGE_SEVERE_BOOST   # Mild 줄이기
        base[2] += AGE_SEVERE_BOOST   # Severe 늘리기

    # 3단계: 생활습관 보정 — 흡연과 음주를 둘 다 하면 Severe 추가 +15%
    if row['smoker'] == 'Yes' and row['alcohol_use'] == 'Yes':
        base[0] -= LIFESTYLE_BOOST
        base[2] += LIFESTYLE_BOOST

    # 4단계: 약물 보정 — 약물별 위험도 반영
    boost = DRUG_SEVERE_BOOST.get(row['drug_name'], 0.0)
    base[0] -= boost   # Mild 감소
    base[2] += boost   # Severe 증가

    # 5단계: 음수가 된 확률을 0으로 바꾸기 (확률은 0 이상이어야 함)
    base = [max(0.0, p) for p in base]
    # 세 값의 합이 1.0이 되도록 정규화
    total = sum(base)
    base  = [p / total for p in base]

    # 6단계: 최종 확률로 Mild/Moderate/Severe 중 하나 랜덤 선택
    # 예: base = [0, 0.17, 0.83] 이면 83% 확률로 Severe 선택
    return rng.choice(['Mild', 'Moderate', 'Severe'], p=base)


# ── 실행 ─────────────────────────────────────────────────────
print("원본 데이터 로드 중...")
df = pd.read_csv(INPUT)
print(f"  {df.shape[0]}행 × {df.shape[1]}열")

print("\n원본 중증도 분포:")
print(df['severity'].value_counts())

print("\n상관관계 주입 중...")
# 각 행마다 assign_smoker 함수를 적용해 smoker 컬럼 값을 새로 덮어씀
df['smoker']      = df.apply(assign_smoker,   axis=1)
# 각 행마다 assign_alcohol 함수를 적용해 alcohol_use 컬럼 값을 새로 덮어씀
df['alcohol_use'] = df.apply(assign_alcohol,  axis=1)
# 각 행마다 assign_severity 함수를 적용해 severity 컬럼 값을 새로 덮어씀
df['severity']    = df.apply(assign_severity, axis=1)

print("\n증강 후 중증도 분포:")
print(df['severity'].value_counts())

# 주입 후 기저질환별 흡연율 확인 (의학적으로 올바르게 됐는지 검증)
print("\n증강 후 기저질환별 흡연율:")
print((df.groupby('chronic_condition')['smoker']
         .apply(lambda x: (x == 'Yes').mean() * 100)
         .round(1)
         .to_string()))

# 최종 결과를 새 CSV 파일로 저장 (train.py가 이 파일을 사용함)
df.to_csv(OUTPUT, index=False)
print(f"\n저장 완료: {OUTPUT}")
