import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# data/ 폴더 경로 설정 (CSV 파일이 저장된 폴더)
DATA_PATH = 'data/'

def load_data():
    # data/ 폴더에서 .csv 파일 목록을 가져옴
    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("data/ 폴더에 CSV 파일이 없습니다.")
    # 첫 번째 CSV 파일의 경로 조합
    path = os.path.join(DATA_PATH, csv_files[0])
    print(f"파일 로드: {path}")
    # pandas로 CSV 읽어서 DataFrame 반환
    return pd.read_csv(path)

# 데이터 불러오기
df = load_data()

# ── 기본 정보 출력 ────────────────────────────────────────────
print("=== 기본 정보 ===")
# df.shape = (행 수, 열 수) 출력
print(f"Shape: {df.shape}")

print("\n컬럼명 & 타입:")
# 각 컬럼 이름과 데이터 타입(int, float, object 등) 출력
for col in df.columns:
    print(f"  - {col!r:30s} ({df[col].dtype})")

# ── 결측값 확인 ────────────────────────────────────────────────
print("\n=== 결측값 ===")
# 각 컬럼별 결측값(NaN) 개수 세기
missing = df.isnull().sum()
# 결측값이 1개 이상인 컬럼만 출력, 없으면 "결측값 없음" 출력
print(missing[missing > 0] if missing.any() else "결측값 없음")

# ── 수치형 컬럼 통계 요약 ─────────────────────────────────────
print("\n=== 수치형 통계 ===")
# count(개수), mean(평균), std(표준편차), min, max 등 자동 계산
print(df.describe())

# ── target 컬럼(severity) 자동 탐지 및 분포 시각화 ───────────
# 컬럼명에 'severity'가 포함된 컬럼을 target으로 사용
target_candidates = [c for c in df.columns if 'severity' in c.lower()]
if target_candidates:
    target_col = target_candidates[0]
    print(f"\n=== '{target_col}' 클래스 분포 ===")
    # 각 클래스(Mild, Moderate, Severe)별 행 수 출력
    print(df[target_col].value_counts())

    # 막대그래프 생성
    plt.figure(figsize=(6, 4))
    counts = df[target_col].value_counts()
    # 초록(경증), 주황(중등도), 빨강(중증) 순서로 색상 지정
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    counts.plot(kind='bar', color=colors[:len(counts)])
    plt.title('Severity 클래스 분포')
    plt.xlabel('Severity')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # 현재 실행 폴더에 PNG로 저장
    plt.savefig('severity_distribution.png')
    print("그래프 저장: severity_distribution.png")
else:
    print("\n⚠️  'severity' 관련 컬럼을 찾지 못했습니다. 위 컬럼명을 확인하고 train.py의 TARGET_COL을 수정하세요.")

print("\nEDA 완료!")
