import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = 'data/'

def load_data():
    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("data/ 폴더에 CSV 파일이 없습니다.")
    path = os.path.join(DATA_PATH, csv_files[0])
    print(f"파일 로드: {path}")
    return pd.read_csv(path)

df = load_data()

print("=== 기본 정보 ===")
print(f"Shape: {df.shape}")
print("\n컬럼명 & 타입:")
for col in df.columns:
    print(f"  - {col!r:30s} ({df[col].dtype})")

print("\n=== 결측값 ===")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "결측값 없음")

print("\n=== 수치형 통계 ===")
print(df.describe())

# Target 컬럼 자동 탐지
target_candidates = [c for c in df.columns if 'severity' in c.lower()]
if target_candidates:
    target_col = target_candidates[0]
    print(f"\n=== '{target_col}' 클래스 분포 ===")
    print(df[target_col].value_counts())

    plt.figure(figsize=(6, 4))
    counts = df[target_col].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    counts.plot(kind='bar', color=colors[:len(counts)])
    plt.title('Severity 클래스 분포')
    plt.xlabel('Severity')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('severity_distribution.png')
    print("그래프 저장: severity_distribution.png")
else:
    print("\n⚠️  'severity' 관련 컬럼을 찾지 못했습니다. 위 컬럼명을 확인하고 train.py의 TARGET_COL을 수정하세요.")

print("\nEDA 완료!")
