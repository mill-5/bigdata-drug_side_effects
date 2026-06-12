import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

TARGET_COL = 'severity'
CONDITIONS = ['Asthma', 'Diabetes', 'Heart Disease', 'Hypertension', 'Kidney Disease']
DRUGS      = ['Amlodipine', 'Amoxicillin', 'Atorvastatin', 'Ibuprofen', 'Insulin',
              'Lisinopril', 'Metformin', 'Omeprazole', 'Paracetamol', 'Sertraline']
CAT_COLS   = ['gender', 'country', 'smoker', 'alcohol_use']

COND_COLS  = [f'cond_{c.replace(" ", "_")}' for c in CONDITIONS]
DRUG_COLS  = [f'drug_{d}' for d in DRUGS]
NUM_COLS   = ['age', 'dosage_mg', 'dosage_x_num_conds']

DATA_PATH  = 'data/'
MODEL_PATH = 'models/'


def load_data():
    aug = os.path.join(DATA_PATH, 'drug_side_effects_augmented.csv')
    src = aug if os.path.exists(aug) else next(
        os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith('.csv')
    )
    print(f"데이터 로드: {src}")
    df = pd.read_csv(src)
    print(f"  {df.shape[0]}행 × {df.shape[1]}열")
    return df


def preprocess(df):
    needed = ['age', 'gender', 'country', 'drug_name', 'dosage_mg',
              'chronic_condition', 'smoker', 'alcohol_use', TARGET_COL]
    df = df[needed].dropna().copy()

    # 기저질환 multi-hot
    for cond, col in zip(CONDITIONS, COND_COLS):
        df[col] = (df['chronic_condition'] == cond).astype(int)

    # 약물 multi-hot
    for drug, col in zip(DRUGS, DRUG_COLS):
        df[col] = (df['drug_name'] == drug).astype(int)

    # 상호작용 피처: 복용량 × 활성 기저질환 수
    df['dosage_x_num_conds'] = df['dosage_mg'] * df[COND_COLS].sum(axis=1)

    FEATURE_COLS = NUM_COLS + COND_COLS + DRUG_COLS + CAT_COLS

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', NUM_COLS + COND_COLS + DRUG_COLS),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
    ])

    X = preprocessor.fit_transform(df[FEATURE_COLS])

    target_le = LabelEncoder()
    y = target_le.fit_transform(df[TARGET_COL].astype(str))

    return X, y, target_le, preprocessor, FEATURE_COLS


def print_dist(label, y, classes):
    print(f"\n{label}")
    for cls, cnt in zip(classes, np.bincount(y)):
        print(f"  {cls}: {cnt}")


def evaluate(label, model, X_test, y_test, classes):
    y_pred = model.predict(X_test)
    print(f"\n=== {label} ===")
    print(classification_report(y_test, y_pred, target_names=classes))
    print(f"Macro F1:    {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       index=classes, columns=classes))
    return f1_score(y_test, y_pred, average='macro')


def train():
    os.makedirs(MODEL_PATH, exist_ok=True)

    df = load_data()
    X, y, target_le, preprocessor, feature_cols = preprocess(df)
    classes = target_le.classes_

    print_dist("원본 클래스 분포:", y, classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1단계: 클래스 가중치 조정
    m_weighted = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    )
    print("\n[1] 클래스 가중치 조정 모델 학습 중...")
    m_weighted.fit(X_train, y_train)
    f1_w = evaluate("클래스 가중치 조정", m_weighted, X_test, y_test, classes)

    # 2단계: SMOTE
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print_dist("SMOTE 후 클래스 분포:", y_res, classes)

    m_smote = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    print("\n[2] SMOTE 모델 학습 중...")
    m_smote.fit(X_res, y_res)
    f1_s = evaluate("SMOTE", m_smote, X_test, y_test, classes)

    best_model = m_weighted if f1_w >= f1_s else m_smote
    best_label = "클래스 가중치 조정" if f1_w >= f1_s else "SMOTE"
    print(f"\n최종 선택: {best_label} (macro F1: {max(f1_w, f1_s):.4f})")

    joblib.dump(best_model,   os.path.join(MODEL_PATH, 'rf_model.pkl'))
    joblib.dump(target_le,    os.path.join(MODEL_PATH, 'target_le.pkl'))
    joblib.dump(preprocessor, os.path.join(MODEL_PATH, 'preprocessor.pkl'))
    joblib.dump(feature_cols, os.path.join(MODEL_PATH, 'feature_cols.pkl'))

    meta = {'conditions': CONDITIONS, 'drugs': DRUGS,
            'cond_cols': COND_COLS, 'drug_cols': DRUG_COLS,
            'num_cols': NUM_COLS, 'cat_cols': CAT_COLS}
    joblib.dump(meta, os.path.join(MODEL_PATH, 'meta.pkl'))
    print("모델 저장 완료: models/rf_model.pkl")


if __name__ == '__main__':
    train()
