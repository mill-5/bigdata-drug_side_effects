import streamlit as st      # 웹 앱을 만드는 라이브러리 (HTML/CSS 없이 Python만으로 UI 구성)
import pandas as pd           # 데이터프레임(표 형태 데이터) 처리
import numpy as np            # 수치 계산 (argmax, argsort 등)
import matplotlib.pyplot as plt  # 그래프 그리기
import matplotlib
import seaborn as sns         # matplotlib 기반 고급 시각화 (히트맵 등)
import joblib                 # 학습된 모델 파일(.pkl) 저장/불러오기
import os                     # 파일 경로, 존재 여부 확인

# 한글 폰트 설정 (Windows 기본 폰트 '맑은 고딕')
# 이 설정 없으면 그래프에서 한글이 깨져서 네모(□)로 표시됨
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 기호 깨짐 방지

# 모델과 데이터 파일이 저장된 폴더 경로
MODEL_PATH = 'models/'
DATA_PATH  = 'data/'

# 브라우저 탭 제목, 아이콘, 레이아웃 설정
# layout='wide': 화면 전체 너비 사용
st.set_page_config(page_title="약물 부작용 예측", page_icon="💊", layout="wide")

# ── 데이터 / 모델 로드 ────────────────────────────────────────

# @st.cache_data: 한 번 실행 후 결과를 메모리에 저장
# Streamlit은 사용자 입력마다 전체 코드를 재실행하는데,
# 캐시 덕분에 100,000행 CSV를 매번 다시 읽지 않아도 됨 → 속도 빠름
@st.cache_data
def load_data():
    # augmented(상관관계 주입된) 파일이 있으면 그것을 우선 사용
    aug = os.path.join(DATA_PATH, 'drug_side_effects_augmented.csv')
    if os.path.exists(aug):
        return pd.read_csv(aug)
    # 없으면 data/ 폴더의 첫 번째 CSV 파일 사용
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    return pd.read_csv(os.path.join(DATA_PATH, files[0]))

# @st.cache_resource: 모델처럼 무거운 객체를 캐싱할 때 사용
# cache_data와 달리 메모리 공유 방식으로 모델 객체를 한 번만 로드함
@st.cache_resource
def load_model():
    # train.py가 저장한 파일들을 불러옴
    model        = joblib.load(os.path.join(MODEL_PATH, 'rf_model.pkl'))       # Random Forest 모델
    target_le    = joblib.load(os.path.join(MODEL_PATH, 'target_le.pkl'))      # 클래스 이름 디코더 (0→Mild, 1→Moderate, 2→Severe)
    preprocessor = joblib.load(os.path.join(MODEL_PATH, 'preprocessor.pkl'))  # 전처리기 (OHE 등)
    feature_cols = joblib.load(os.path.join(MODEL_PATH, 'feature_cols.pkl'))  # 특성 컬럼 목록
    meta         = joblib.load(os.path.join(MODEL_PATH, 'meta.pkl'))           # 약물/기저질환 목록 메타정보
    return model, target_le, preprocessor, feature_cols, meta

# ── 한국어 매핑 ───────────────────────────────────────────────
# 영어 → 한국어 변환 딕셔너리들
# 앱 화면에서 영어 대신 한국어로 표시하기 위해 사용

CONDITION_KO = {
    'Asthma': '천식', 'Diabetes': '당뇨병',
    'Heart Disease': '심장질환', 'Hypertension': '고혈압',
    'Kidney Disease': '신장질환',
}
DRUG_KO = {
    'Amlodipine': '암로디핀', 'Amoxicillin': '아목시실린',
    'Atorvastatin': '아토르바스타틴', 'Ibuprofen': '이부프로펜',
    'Insulin': '인슐린', 'Lisinopril': '리시노프릴',
    'Metformin': '메트포르민', 'Omeprazole': '오메프라졸',
    'Paracetamol': '파라세타몰', 'Sertraline': '설트랄린',
}
SEVERITY_KO = {'Mild': '경증', 'Moderate': '중등도', 'Severe': '중증'}
SIDE_EFFECT_KO = {
    'Muscle Pain': '근육통', 'Dry Mouth': '구강건조', 'Hypoglycemia': '저혈당',
    'Rash': '발진', 'Dizziness': '어지러움', 'Nausea': '구역질',
    'Sweating': '발한', 'Swelling': '부종', 'Anxiety': '불안감',
    'Stomach Pain': '위통', 'Abdominal Pain': '복통', 'Headache': '두통',
    'Fatigue': '피로감', 'Weight Gain': '체중 증가', 'Heartburn': '속쓰림',
    'Diarrhea': '설사', 'Dry Cough': '건성 기침', 'Constipation': '변비',
    'Liver Toxicity': '간독성', 'Insomnia': '불면증', 'Blurred Vision': '시야 흐림',
    'Joint Pain': '관절통', 'Back Pain': '요통', 'Chest Pain': '흉통',
    'Vomiting': '구토', 'Palpitations': '두근거림', 'Edema': '부종',
}

# ── 사이드바 네비게이션 ───────────────────────────────────────
# 왼쪽 사이드바에 라디오 버튼 메뉴 생성
# 선택한 값이 page 변수에 저장되고, 아래 if/elif로 해당 페이지를 보여줌
page = st.sidebar.radio(
    "메뉴",
    ["💊 부작용 예측", "🔍 EDA", "📊 데이터 분석", "⚙️ 전처리·특성 엔지니어링"],
    label_visibility="collapsed"  # "메뉴" 라벨 텍스트를 화면에서 숨김
)

# ════════════════════════════════════════════════════════════
# 페이지 1: 부작용 예측
# ════════════════════════════════════════════════════════════
if page == "💊 부작용 예측":

    def plot_proba(classes, proba):
        # 예측 확률을 가로 막대그래프로 시각화하는 함수
        # classes: ['Mild', 'Moderate', 'Severe']
        # proba: [0.05, 0.15, 0.80] 같은 확률 배열
        label_map  = {c: f'{c}\n({SEVERITY_KO.get(c, c)})' for c in classes}
        bar_colors = ['#2ecc71', '#f39c12', '#e74c3c']  # 초록, 주황, 빨강
        labels     = [label_map[c] for c in classes]
        fig, ax = plt.subplots(figsize=(5, 3))
        # barh = 가로 막대그래프, proba*100으로 % 단위로 변환
        bars = ax.barh(labels, proba * 100, color=bar_colors, edgecolor='white', height=0.5)
        # 각 막대 끝에 퍼센트 숫자 표시
        for bar, p in zip(bars, proba):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{p*100:.1f}%', va='center', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 115)   # x축 범위 (숫자 라벨 공간 확보)
        ax.set_xlabel('확률 (%)', fontsize=10)
        ax.set_title('부작용 중증도 예측 확률', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)    # 위쪽 테두리 제거 (깔끔한 디자인)
        ax.spines['right'].set_visible(False)  # 오른쪽 테두리 제거
        plt.tight_layout()
        return fig

    def plot_feature_importance(model, preprocessor):
        # Random Forest 모델의 피처 중요도 Top 12를 가로 막대그래프로 시각화
        from train import CONDITIONS, DRUGS, COND_COLS, DRUG_COLS, NUM_COLS, CAT_COLS
        # 전처리기에서 One-Hot Encoding된 컬럼 이름 가져오기
        ohe       = preprocessor.named_transformers_['cat']
        passthrough_names = NUM_COLS + COND_COLS + DRUG_COLS  # 숫자형 컬럼명
        cat_names = ohe.get_feature_names_out(CAT_COLS).tolist()  # OHE 컬럼명
        all_names = passthrough_names + cat_names  # 전체 컬럼명 리스트

        # 영어 컬럼명 → 한국어 라벨 매핑
        cond_ko = {f'cond_{c.replace(" ","_")}': CONDITION_KO.get(c, c) for c in CONDITIONS}
        drug_ko = {f'drug_{d}': DRUG_KO.get(d, d) for d in DRUGS}
        label_ko = {
            'age': '나이', 'dosage_mg': '복용량', 'dosage_x_num_conds': '복용량×질환수',
            'gender_Male': '성별(남)', 'gender_Female': '성별(여)',
            'smoker_Yes': '흡연', 'smoker_No': '비흡연',
            'alcohol_use_Yes': '음주', 'alcohol_use_No': '비음주',
            **cond_ko, **drug_ko,
        }
        def shorten(n):
            # 컬럼명을 한국어로 변환, 국가 컬럼은 'country_' 접두사만 제거
            if n in label_ko: return label_ko[n]
            for p in ['country_']:
                if n.startswith(p): return n.replace(p, '')
            return n

        # model.feature_importances_: 각 특성이 예측에 얼마나 기여했는지 (0~1, 합=1)
        imp = model.feature_importances_
        # argsort로 중요도 순 정렬 후 상위 12개 인덱스 선택
        idx = np.argsort(imp)[-12:]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh([shorten(all_names[i]) for i in idx], imp[idx],
                color='#3498db', edgecolor='white', height=0.6)
        ax.set_xlabel('중요도', fontsize=10)
        ax.set_title('피처 중요도 Top 12', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig

    st.title("💊 약물 부작용 중증도 예측")
    st.markdown("환자 정보를 입력하면 부작용 중증도를 예측합니다.")
    st.divider()

    # 화면을 좌우 2개 컬럼으로 나눔
    col1, col2 = st.columns(2)
    with col1:
        age     = st.number_input("나이", min_value=1, max_value=120, value=40)
        gender  = st.selectbox("성별", ["Male", "Female"])
        country = st.selectbox("국가", ['Australia', 'Canada', 'Germany', 'India', 'Pakistan', 'UK', 'USA'])
        # 약물 옵션: "Insulin (인슐린)": "Insulin" 형태의 딕셔너리
        drug_options = {f'{e} ({k})': e for e, k in DRUG_KO.items()}
        drugs_selected = st.multiselect(
            "복용 약물 (복수 선택 가능)",
            list(drug_options.keys()),
            default=[list(drug_options.keys())[0]]  # 첫 번째 약물을 기본 선택
        )
    with col2:
        dosage = st.number_input("복용량 (mg)", min_value=0.0, value=100.0)
        # 기저질환 옵션: 영어-한국어 딕셔너리
        cond_options = {f'{e} ({k})': e for e, k in CONDITION_KO.items()}
        COND_NONE = "없음 (기저질환 없음)"  # 기저질환 없음 옵션
        conds_selected = st.multiselect(
            "기저질환 (복수 선택 가능)",
            [COND_NONE] + list(cond_options.keys()),  # 없음을 맨 위에 배치
            default=[COND_NONE]  # 기본값: 없음
        )
        smoking = st.selectbox("흡연 여부", ["No", "Yes"])
        alcohol_freq = st.slider("음주 횟수 (월)", min_value=0, max_value=30, value=0,
                                  help="한 달에 음주하는 횟수를 선택하세요 (0 = 음주 안 함)")
        # 월 음주 횟수가 1회 이상이면 Yes, 0이면 No로 변환
        alcohol = "Yes" if alcohol_freq > 0 else "No"

    st.divider()

    # 예측하기 버튼 — 클릭 시 아래 블록 실행
    if st.button("예측하기", type="primary", use_container_width=True):
        # 모델 파일 존재 여부 확인 (train.py 실행 전이면 파일 없음)
        if not os.path.exists(os.path.join(MODEL_PATH, 'rf_model.pkl')):
            st.error("모델 파일이 없습니다. 먼저 `python train.py`를 실행하세요.")
            st.stop()
        if not drugs_selected:
            st.warning("복용 약물을 최소 1개 선택해주세요.")
            st.stop()
        if not conds_selected:
            st.warning("기저질환을 선택하거나 '없음'을 선택해주세요.")
            st.stop()

        # 모델과 전처리기 불러오기
        model, target_le, preprocessor, feature_cols, meta = load_model()
        classes    = target_le.classes_   # ['Mild', 'Moderate', 'Severe']
        color_map  = {'Mild': '🟢', 'Moderate': '🟡', 'Severe': '🔴'}
        label_map  = {c: f'{c} ({SEVERITY_KO.get(c, c)})' for c in classes}
        severe_idx = list(classes).index('Severe')

        # "없음" 옵션은 걸러내고 실제 기저질환 영어명만 추출
        # cond_options에 없는 키("없음")는 if c in cond_options로 걸러냄
        selected_conds = [cond_options[c] for c in conds_selected if c in cond_options]
        selected_drugs = [drug_options[d] for d in drugs_selected]

        # ── 모델 입력 벡터 구성 ──────────────────────────
        num_conds = len(selected_conds)  # 기저질환 개수 (없음이면 0)
        row = {
            'age': age,
            'gender': gender,
            'country': country,
            'dosage_mg': dosage,
            'smoker': smoking,
            'alcohol_use': alcohol,
            # 핵심 파생 특성: 복용량 × 기저질환 수
            # 기저질환이 많을수록 같은 복용량도 더 위험하다는 의미
            'dosage_x_num_conds': dosage * num_conds,
        }
        # 기저질환 multi-hot: 선택된 기저질환이면 1, 아니면 0
        for cond, col in zip(meta['conditions'], meta['cond_cols']):
            row[col] = int(cond in selected_conds)
        # 약물 multi-hot: 선택된 약물이면 1, 아니면 0
        for drug, col in zip(meta['drugs'], meta['drug_cols']):
            row[col] = int(drug in selected_drugs)

        # 입력 벡터를 DataFrame으로 만든 후 전처리기로 변환 (OHE 적용)
        X_input    = preprocessor.transform(pd.DataFrame([row])[feature_cols])
        # predict_proba: 각 클래스의 확률 반환 (예: [0.05, 0.15, 0.80])
        proba      = model.predict_proba(X_input)[0]
        # argmax: 가장 큰 값의 인덱스 → 가장 높은 확률의 클래스 선택
        pred_label = classes[int(np.argmax(proba))]

        # ── 예측 결과 출력 ─────────────────────────────────────
        st.subheader("예측 결과")
        # 복수 선택 시 선택된 기저질환·약물명 표시
        if len(selected_conds) > 1 or len(selected_drugs) > 1:
            cond_str = ', '.join(f"{c} ({CONDITION_KO.get(c,c)})" for c in selected_conds)
            drug_str = ', '.join(f"{d} ({DRUG_KO.get(d,d)})" for d in selected_drugs)
            st.caption(f"기저질환: **{cond_str}** | 복용 약물: **{drug_str}**")

        # 각 클래스의 확률을 숫자 카드(metric)로 표시
        mc = st.columns(len(classes))
        for i, cls in enumerate(classes):
            with mc[i]:
                st.metric(f"{color_map.get(cls,'⚪')} {label_map[cls]}", f"{proba[i]*100:.1f}%")

        # 예측 결과에 따라 다른 색상의 경고/안내 메시지 표시
        if pred_label == 'Severe':
            st.error("⚠️ **고위험 경고**: 부작용 중증도가 **Severe (중증)**으로 예측됩니다.\n\n즉시 의료진에게 상담하세요.")
        elif pred_label == 'Moderate':
            st.warning("**Moderate (중등도)** 수준의 부작용이 예측됩니다. 의료진과 상담을 권장합니다.")
        else:
            st.success("**Mild (경증)** 수준의 부작용이 예측됩니다.")

        # ── 주요 부작용 증상 ──────────────────────────────────
        st.divider()
        df_ref = load_data()
        # 선택한 약물에 해당하는 행 필터링
        drug_mask = df_ref['drug_name'].isin(selected_drugs)
        # 선택한 기저질환에 해당하는 행 필터링 (없으면 전체)
        cond_mask = (
            df_ref['chronic_condition'].isin(selected_conds)
            if selected_conds else pd.Series(True, index=df_ref.index)
        )
        df_sub = df_ref[drug_mask & cond_mask]  # 두 조건 모두 만족하는 행

        st.subheader("주요 예상 부작용 증상")
        if 'side_effect' in df_sub.columns and len(df_sub) >= 5:
            # 가장 많이 나타나는 부작용 Top 5 (비율로 정규화)
            top_effects = df_sub['side_effect'].value_counts(normalize=True).head(5)
            for effect, pct in top_effects.items():
                ko = SIDE_EFFECT_KO.get(effect, '')
                label = f"{effect} ({ko})" if ko else effect
                st.markdown(f"**{label}** &nbsp; `{pct*100:.1f}%`")
                st.progress(float(pct))  # 비율을 진행 바로 표시
            st.caption(f"유사 환자 {len(df_sub):,}명 기준")
        else:
            st.info("해당 조합의 데이터가 부족합니다.")

        # ── 맞춤 건강 권고사항 ────────────────────────────────
        st.divider()
        st.subheader("맞춤 건강 권고사항")
        tips = []  # (아이콘, 제목, 설명) 튜플 리스트

        # 입력 조건에 따라 관련 건강 팁 추가
        if smoking == 'Yes':
            tips.append(("🚭", "금연", "흡연은 모든 기저질환에서 부작용 중증도를 높입니다. 금연 시 Severe 위험이 평균 15%p 감소합니다."))
        if alcohol_freq >= 8:
            tips.append(("🍷", "금주 권장", f"월 {alcohol_freq}회 음주는 고위험 수준입니다. 음주는 약물 대사를 심각하게 방해하며 부작용을 악화시킵니다. 특히 Metformin, Sertraline 복용 중에는 즉시 금주를 권장합니다."))
        elif alcohol_freq >= 1:
            tips.append(("🍷", "절주 권장", f"현재 월 {alcohol_freq}회 음주 중입니다. 음주는 약물 대사를 방해하고 부작용을 악화시킬 수 있습니다. 복용 약물에 따라 금주 또는 절주를 권장합니다."))
        if age >= 60:
            tips.append(("👴", "정기적인 건강검진", "60세 이상은 약물 대사 능력이 저하되어 동일 복용량에서도 부작용이 강하게 나타날 수 있습니다. 3~6개월 주기로 검진을 받으세요."))
        if 'Heart Disease' in selected_conds or 'Hypertension' in selected_conds:
            tips.append(("🏃", "규칙적인 유산소 운동", "심장질환·고혈압 환자는 주 3회 이상 30분 걷기·수영 등 저강도 유산소 운동이 혈압 조절과 약물 효과 향상에 도움이 됩니다."))
        if 'Diabetes' in selected_conds:
            tips.append(("🥗", "저탄수화물 식단 관리", "당뇨 환자는 혈당 급등을 막기 위해 정제 탄수화물을 줄이고 식이섬유를 늘리세요. 식후 30분 가벼운 산책도 혈당 안정에 효과적입니다."))
        if 'Kidney Disease' in selected_conds:
            tips.append(("💧", "수분 섭취 및 저단백 식단", "신장질환 환자는 하루 1.5~2L 수분 섭취를 유지하고, 단백질 과다 섭취를 피해 신장 부담을 줄이세요."))
        if 'Asthma' in selected_conds:
            tips.append(("🌬️", "실내 공기질 관리", "천식 환자는 미세먼지·꽃가루 노출을 최소화하고, 환기 시 공기청정기를 활용하세요. Ibuprofen은 천식을 악화시킬 수 있으므로 의사와 상의하세요."))
        if pred_label in ('Moderate', 'Severe'):
            tips.append(("💊", "복용량 및 약물 재검토", "현재 복용량과 약물 조합이 중등도 이상의 부작용을 유발할 수 있습니다. 반드시 담당 의사와 복용량 조정을 상담하세요."))
        if len(selected_drugs) >= 2:
            tips.append(("⚠️", "다약제 상호작용 주의", f"{len(selected_drugs)}가지 약물을 동시 복용 중입니다. 약물 간 상호작용이 부작용을 강화할 수 있으므로 약사 또는 의사에게 병용 적합성을 확인하세요."))

        if not tips:
            st.success("현재 입력된 정보 기준으로 특별한 위험 요인이 발견되지 않았습니다. 규칙적인 복약과 건강한 생활습관을 유지하세요.")
        else:
            # 각 팁을 테두리 있는 카드 형태로 표시
            for icon, title, desc in tips:
                with st.container(border=True):
                    st.markdown(f"**{icon} {title}**")
                    st.caption(desc)

        st.divider()
        st.subheader("시각화")
        # 화면을 좌우로 나눠 두 그래프 나란히 표시
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_proba(classes, proba))           # 확률 막대그래프
        with c2:
            st.pyplot(plot_feature_importance(model, preprocessor))  # 피처 중요도


# ════════════════════════════════════════════════════════════
# 페이지 2: EDA
# ════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("🔍 탐색적 데이터 분석 (EDA)")
    st.markdown("모델 학습에 사용된 데이터의 구조와 분포를 확인합니다.")
    st.divider()

    df = load_data()

    # ── 1. 데이터 개요 ───────────────────────────────────────
    st.subheader("1. 데이터 개요")
    # 3개 컬럼에 숫자 카드(metric) 표시
    c1, c2, c3 = st.columns(3)
    c1.metric("전체 행 수", f"{df.shape[0]:,}")
    c2.metric("컬럼 수",    f"{df.shape[1]}")
    c3.metric("결측값",     f"{df.isnull().sum().sum()}")  # 전체 결측값 합계

    # 클릭하면 펼쳐지는 상세 정보
    with st.expander("컬럼 목록 및 타입 보기"):
        info = pd.DataFrame({
            '컬럼명': df.columns,
            '타입':   df.dtypes.astype(str).values,
            '결측값': df.isnull().sum().values,
            '고유값 수': df.nunique().values,  # 각 컬럼에 몇 종류의 값이 있는지
        })
        st.dataframe(info, use_container_width=True, hide_index=True)

    st.divider()

    # ── 2. 타깃 변수 분포 ────────────────────────────────────
    st.subheader("2. 타깃 변수 분포 (severity)")
    # Mild → Moderate → Severe 순서로 정렬
    sev_counts = df['severity'].value_counts().reindex(['Mild', 'Moderate', 'Severe'])
    col1, col2 = st.columns([1, 2])
    with col1:
        # 표 형태로 건수와 비율 표시
        st.dataframe(
            sev_counts.rename_axis('중증도').reset_index(name='건수').assign(
                비율=lambda x: (x['건수'] / x['건수'].sum() * 100).round(1).astype(str) + '%'
            ),
            use_container_width=True, hide_index=True
        )
    with col2:
        # 막대그래프 생성
        fig_s, ax_s = plt.subplots(figsize=(6, 3))
        colors_s = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax_s.bar(sev_counts.index, sev_counts.values, color=colors_s, edgecolor='white', width=0.5)
        # 막대 위에 숫자 표시
        for bar, v in zip(bars, sev_counts.values):
            ax_s.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                      f'{v:,}', ha='center', fontsize=10, fontweight='bold')
        ax_s.set_ylabel('환자 수'); ax_s.set_title('Severity 클래스 분포', fontsize=12, fontweight='bold')
        ax_s.spines['top'].set_visible(False); ax_s.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_s)

    st.divider()

    # ── 3. 수치형 변수 분포 ──────────────────────────────────
    st.subheader("3. 수치형 변수 분포")
    # 나이와 복용량의 기술통계 (평균, 표준편차, 최솟값, 최댓값 등)
    st.dataframe(df[['age', 'dosage_mg']].describe().round(2), use_container_width=True)

    # 나이와 복용량 히스토그램 (분포 형태 확인)
    fig_n, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, col, color, label in zip(
        axes,
        ['age', 'dosage_mg'],
        ['#3498db', '#9b59b6'],
        ['나이 (age)', '복용량 (dosage_mg)']
    ):
        ax.hist(df[col].dropna(), bins=30, color=color, edgecolor='white', alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('환자 수')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_n)

    st.divider()

    # ── 4. 범주형 변수 분포 ──────────────────────────────────
    st.subheader("4. 범주형 변수 분포")
    cat_targets = {
        'gender':            '성별',
        'country':           '국가',
        'chronic_condition': '기저질환',
        'drug_name':         '복용 약물',
        'smoker':            '흡연 여부',
        'alcohol_use':       '음주 여부',
    }
    # 3열 그리드로 6개 그래프 배치
    cols_cat = st.columns(3)
    for idx, (col, title) in enumerate(cat_targets.items()):
        with cols_cat[idx % 3]:  # 0→첫째, 1→둘째, 2→셋째, 3→첫째 순환
            vc = df[col].value_counts()
            fig_c, ax_c = plt.subplots(figsize=(4, 3))
            ax_c.barh(vc.index, vc.values, color='#1abc9c', edgecolor='white', height=0.6)
            ax_c.set_title(title, fontsize=11, fontweight='bold')
            ax_c.spines['top'].set_visible(False); ax_c.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_c)

    st.divider()

    # ── 5. 상관관계 히트맵 ───────────────────────────────────
    st.subheader("5. 수치형 변수 상관관계")
    st.caption("나이·복용량과 중증도(숫자 인코딩) 간의 상관계수를 확인합니다.")
    from sklearn.preprocessing import LabelEncoder
    df_corr = df[['age', 'dosage_mg', 'severity']].copy()
    # severity는 문자열이므로 숫자로 변환 (Mild=0, Moderate=1, Severe=2)
    df_corr['severity_enc'] = LabelEncoder().fit_transform(df_corr['severity'])
    corr = df_corr[['age', 'dosage_mg', 'severity_enc']].rename(
        columns={'age': '나이', 'dosage_mg': '복용량', 'severity_enc': '중증도'}
    ).corr()  # 피어슨 상관계수 계산 (-1~1, 1에 가까울수록 양의 상관관계)

    fig_h, ax_h = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                linewidths=0.5, ax=ax_h, vmin=-1, vmax=1,
                cbar_kws={'label': '상관계수'})
    ax_h.set_title('수치형 변수 상관관계 히트맵', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_h)

    st.divider()

    # ── 6. 이상값 탐지 (박스플롯) ────────────────────────────
    st.subheader("6. 이상값 탐지")
    st.caption("나이·복용량의 분포와 이상값을 중증도별로 비교합니다.")
    fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(12, 4))
    order_sev = ['Mild', 'Moderate', 'Severe']
    colors_b  = ['#2ecc71', '#f39c12', '#e74c3c']
    for ax_b, col, label in zip([ax_b1, ax_b2], ['age', 'dosage_mg'], ['나이 (age)', '복용량 (dosage_mg)']):
        data_by_sev = [df[df['severity'] == s][col].dropna() for s in order_sev]
        # 박스플롯: 중앙값, 사분위수, 이상값(점)을 한 번에 표시
        bp = ax_b.boxplot(data_by_sev, patch_artist=True, labels=['경증', '중등도', '중증'],
                          medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors_b):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax_b.set_title(label, fontsize=11, fontweight='bold')
        ax_b.set_ylabel(col)
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_b)

    st.divider()

    # ── 7. 원본 데이터 샘플 ──────────────────────────────────
    st.subheader("7. 원본 데이터 샘플")
    n = st.slider("표시할 행 수", min_value=5, max_value=50, value=10, step=5)
    st.dataframe(df.head(n), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# 페이지 3: 데이터 분석
# ════════════════════════════════════════════════════════════
elif page == "📊 데이터 분석":
    st.title("📊 데이터 분석")
    st.markdown("100,000건의 약물 부작용 데이터를 탐색합니다.")
    st.divider()

    df = load_data()
    # 영어 값을 한국어로 변환한 컬럼 추가 (그래프 라벨용)
    df['condition_ko'] = df['chronic_condition'].map(CONDITION_KO)
    df['drug_ko']      = df['drug_name'].map(DRUG_KO)
    df['severity_ko']  = df['severity'].map(SEVERITY_KO)

    # 4개 탭으로 분석 분류
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏥 기저질환 × 흡연·음주",
        "🌍 국가별 기저질환·약물",
        "💊 약물별 중증도",
        "👤 나이·성별 분포",
    ])

    # ── 탭1: 기저질환 × 흡연·음주 ────────────────────────────
    with tab1:
        st.subheader("기저질환별 흡연·음주 비율")
        st.caption("각 기저질환 환자 중 흡연자·음주자의 비율을 비교합니다.")

        # groupby로 기저질환별 흡연율 계산 (Yes인 비율 × 100)
        smoke_rate = df.groupby('condition_ko')['smoker'].apply(
            lambda x: (x == 'Yes').mean() * 100).reset_index(name='흡연율(%)')
        alc_rate   = df.groupby('condition_ko')['alcohol_use'].apply(
            lambda x: (x == 'Yes').mean() * 100).reset_index(name='음주율(%)')
        # 흡연율과 음주율 테이블 합치기
        merged = smoke_rate.merge(alc_rate, on='condition_ko').rename(columns={'condition_ko': '기저질환'})

        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(merged))
        w = 0.35  # 막대 너비
        # 나란히 배치되는 이중 막대그래프
        b1 = ax.bar(x - w/2, merged['흡연율(%)'], w, label='흡연율', color='#e74c3c', alpha=0.85)
        b2 = ax.bar(x + w/2, merged['음주율(%)'], w, label='음주율', color='#3498db', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(merged['기저질환'], fontsize=11)
        ax.set_ylabel('비율 (%)'); ax.set_ylim(0, 100)
        ax.set_title('기저질환별 흡연·음주 비율', fontsize=13, fontweight='bold')
        ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        # 막대 위에 수치 표시
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f'{b.get_height():.1f}%', ha='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("기저질환별 중증도 분포")
        st.caption("각 기저질환 환자가 어떤 중증도 비율을 보이는지 확인합니다.")
        # crosstab으로 기저질환 × 중증도 빈도표 생성
        sev_ct = df.groupby(['condition_ko', 'severity_ko']).size().unstack(fill_value=0)
        # 절대값 → 비율(%)로 변환
        sev_pct = sev_ct.div(sev_ct.sum(axis=1), axis=0) * 100
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        # stacked=True: 누적 막대그래프 (합이 100%가 되도록)
        sev_pct[['경증', '중등도', '중증']].plot(kind='bar', stacked=True, ax=ax2,
                                                  color=colors, edgecolor='white')
        ax2.set_xlabel(''); ax2.set_ylabel('비율 (%)'); ax2.set_ylim(0, 110)
        ax2.set_title('기저질환별 중증도 비율', fontsize=13, fontweight='bold')
        ax2.legend(title='중증도', bbox_to_anchor=(1.01, 1))
        ax2.tick_params(axis='x', rotation=0)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)

    # ── 탭2: 국가별 기저질환·약물 ────────────────────────────
    with tab2:
        st.subheader("국가별 기저질환 분포 (히트맵)")
        st.caption("나라마다 어떤 기저질환 환자가 많은지 비교합니다.")
        # crosstab: 행=국가, 열=기저질환, 값=환자 수
        ct_cond = pd.crosstab(df['country'], df['condition_ko'])
        # 각 국가 내 비율(%)로 변환
        ct_cond_pct = ct_cond.div(ct_cond.sum(axis=1), axis=0) * 100
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        # annot=True: 셀 안에 수치 표시, fmt='.1f': 소수점 1자리
        sns.heatmap(ct_cond_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax3, cbar_kws={'label': '비율 (%)'})
        ax3.set_title('국가별 기저질환 비율 (%)', fontsize=13, fontweight='bold')
        ax3.set_xlabel(''); ax3.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig3)

        st.subheader("국가별 처방 약물 분포 (히트맵)")
        st.caption("나라마다 어떤 약물을 많이 복용하는지 비교합니다.")
        ct_drug = pd.crosstab(df['country'], df['drug_ko'])
        ct_drug_pct = ct_drug.div(ct_drug.sum(axis=1), axis=0) * 100
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        sns.heatmap(ct_drug_pct, annot=True, fmt='.1f', cmap='Blues',
                    linewidths=0.5, ax=ax4, cbar_kws={'label': '비율 (%)'})
        ax4.set_title('국가별 처방 약물 비율 (%)', fontsize=13, fontweight='bold')
        ax4.set_xlabel(''); ax4.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig4)

    # ── 탭3: 약물별 중증도 ───────────────────────────────────
    with tab3:
        st.subheader("약물별 중증도 비율")
        st.caption("약물마다 Severe(중증) 비율이 얼마나 다른지 확인합니다.")
        drug_sev = df.groupby(['drug_ko', 'severity_ko']).size().unstack(fill_value=0)
        drug_sev_pct = drug_sev.div(drug_sev.sum(axis=1), axis=0) * 100
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        drug_sev_pct[['경증', '중등도', '중증']].plot(kind='bar', stacked=True, ax=ax5,
                                                       color=['#2ecc71', '#f39c12', '#e74c3c'],
                                                       edgecolor='white')
        ax5.set_xlabel(''); ax5.set_ylabel('비율 (%)'); ax5.set_ylim(0, 115)
        ax5.set_title('약물별 중증도 비율', fontsize=13, fontweight='bold')
        ax5.legend(title='중증도', bbox_to_anchor=(1.01, 1))
        ax5.tick_params(axis='x', rotation=30)
        ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5)

    # ── 탭4: 나이·성별 분포 ──────────────────────────────────
    with tab4:
        st.subheader("중증도별 나이 분포")
        st.caption("중증도가 높을수록 나이대가 다른지 확인합니다.")
        fig6, ax6 = plt.subplots(figsize=(8, 4))
        order = ['경증', '중등도', '중증']
        colors6 = ['#2ecc71', '#f39c12', '#e74c3c']
        # 중증도별로 나이 히스토그램을 겹쳐서 그림 (alpha=0.6으로 투명도 적용)
        for sev, color in zip(order, colors6):
            subset = df[df['severity_ko'] == sev]['age']
            ax6.hist(subset, bins=20, alpha=0.6, label=sev, color=color, edgecolor='white')
        ax6.set_xlabel('나이'); ax6.set_ylabel('환자 수')
        ax6.set_title('중증도별 나이 분포', fontsize=13, fontweight='bold')
        ax6.legend(title='중증도')
        ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig6)

        st.subheader("성별 × 중증도 분포")
        st.caption("남성과 여성의 중증도 비율 차이를 비교합니다.")
        gender_sev = df.groupby(['gender', 'severity_ko']).size().unstack(fill_value=0)
        gender_sev_pct = gender_sev.div(gender_sev.sum(axis=1), axis=0) * 100
        fig7, ax7 = plt.subplots(figsize=(5, 4))
        gender_sev_pct[['경증', '중등도', '중증']].plot(kind='bar', stacked=True, ax=ax7,
                                                         color=['#2ecc71', '#f39c12', '#e74c3c'],
                                                         edgecolor='white')
        ax7.set_xlabel(''); ax7.set_ylabel('비율 (%)'); ax7.set_ylim(0, 115)
        ax7.set_title('성별 × 중증도 비율', fontsize=13, fontweight='bold')
        ax7.legend(title='중증도', bbox_to_anchor=(1.01, 1))
        ax7.tick_params(axis='x', rotation=0)
        ax7.spines['top'].set_visible(False); ax7.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig7)


# ════════════════════════════════════════════════════════════
# 페이지 4: 전처리·특성 엔지니어링
# ════════════════════════════════════════════════════════════
elif page == "⚙️ 전처리·특성 엔지니어링":
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE

    st.title("⚙️ 전처리 및 특성 엔지니어링")
    st.markdown("원본 데이터를 모델 입력으로 변환하는 과정을 단계별로 설명합니다.")
    st.divider()

    df_raw = load_data()

    # 4개 탭으로 전처리 과정 단계별 설명
    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ 결측값 처리",
        "2️⃣ 인코딩",
        "3️⃣ 특성 엔지니어링",
        "4️⃣ 클래스 불균형 처리",
    ])

    # ── 탭1: 결측값 처리 ─────────────────────────────────────
    with tab1:
        st.subheader("결측값 현황")
        missing = df_raw.isnull().sum()
        miss_df = pd.DataFrame({
            '컬럼': missing.index,
            '결측값 수': missing.values,
            '결측률 (%)': (missing / len(df_raw) * 100).round(2).values
        })
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

        total_missing = missing.sum()
        if total_missing == 0:
            st.success("결측값 없음 — dropna() 처리 후에도 전체 행이 유지됩니다.")
        else:
            st.warning(f"총 {total_missing}개의 결측값이 있습니다. `dropna()`로 해당 행을 제거합니다.")

        st.subheader("처리 전후 행 수 비교")
        needed = ['age', 'gender', 'country', 'drug_name', 'dosage_mg',
                  'chronic_condition', 'smoker', 'alcohol_use', 'severity']
        df_clean = df_raw[needed].dropna()  # 필요한 컬럼만 선택 후 결측값 행 제거
        c1, c2, c3 = st.columns(3)
        c1.metric("처리 전", f"{len(df_raw):,}행")
        c2.metric("처리 후", f"{len(df_clean):,}행")
        c3.metric("제거된 행", f"{len(df_raw) - len(df_clean):,}행")

    # ── 탭2: 인코딩 ──────────────────────────────────────────
    with tab2:
        st.subheader("기저질환 Multi-Hot 인코딩")
        st.caption("기저질환 1개 컬럼 → 5개 이진(0/1) 컬럼으로 변환합니다. 복수 선택 시 여러 컬럼에 1이 설정됩니다.")
        CONDITIONS = ['Asthma', 'Diabetes', 'Heart Disease', 'Hypertension', 'Kidney Disease']
        # 인코딩 예시 데이터 생성
        ex_rows = [
            {'chronic_condition': 'Diabetes',
             **{f'cond_{c.replace(" ","_")}': int(c == 'Diabetes') for c in CONDITIONS}},
            {'chronic_condition': 'Heart Disease',
             **{f'cond_{c.replace(" ","_")}': int(c == 'Heart Disease') for c in CONDITIONS}},
            {'chronic_condition': '(복수: Asthma+Hypertension)',
             **{f'cond_{c.replace(" ","_")}': int(c in ['Asthma', 'Hypertension']) for c in CONDITIONS}},
        ]
        st.dataframe(pd.DataFrame(ex_rows), use_container_width=True, hide_index=True)

        st.subheader("약물 Multi-Hot 인코딩")
        st.caption("약물 1개 컬럼 → 10개 이진(0/1) 컬럼으로 변환합니다.")
        DRUGS = ['Amlodipine', 'Amoxicillin', 'Atorvastatin', 'Ibuprofen', 'Insulin',
                 'Lisinopril', 'Metformin', 'Omeprazole', 'Paracetamol', 'Sertraline']
        ex_drugs = [
            {'drug_name': 'Insulin',
             **{f'drug_{d}': int(d == 'Insulin') for d in DRUGS}},
            {'drug_name': '(복수: Metformin+Insulin)',
             **{f'drug_{d}': int(d in ['Metformin', 'Insulin']) for d in DRUGS}},
        ]
        st.dataframe(pd.DataFrame(ex_drugs), use_container_width=True, hide_index=True)

        st.subheader("나머지 범주형 변수 — One-Hot Encoding")
        st.caption("`gender`, `country`, `smoker`, `alcohol_use`는 OHE를 적용합니다.")
        ohe_ex = pd.DataFrame({
            'gender': ['Male', 'Female'],
            'smoker': ['Yes', 'No'],
            'gender_Female': [0, 1],
            'gender_Male':   [1, 0],
            'smoker_No':     [0, 1],
            'smoker_Yes':    [1, 0],
        })
        st.dataframe(ohe_ex, use_container_width=True, hide_index=True)

    # ── 탭3: 특성 엔지니어링 ─────────────────────────────────
    with tab3:
        st.subheader("상호작용 피처: dosage_x_num_conds")
        st.caption("복용량(mg) × 활성 기저질환 수 → 기저질환이 많을수록 동일 복용량의 위험도가 올라간다는 가정을 반영합니다.")

        # 계산 예시 표
        demo = pd.DataFrame({
            '나이': [45, 62, 38],
            '복용량 (mg)': [100, 200, 50],
            '활성 기저질환 수': [1, 3, 1],
            'dosage_x_num_conds': [100*1, 200*3, 50*1],  # 복용량 × 기저질환 수
        })
        st.dataframe(demo, use_container_width=True, hide_index=True)

        st.subheader("dosage_x_num_conds 분포")
        df_fe = df_raw.copy()
        COND_LIST = ['Asthma', 'Diabetes', 'Heart Disease', 'Hypertension', 'Kidney Disease']
        df_fe['num_conds'] = 1  # 원본 데이터는 기저질환 1개
        df_fe['dosage_x_num_conds'] = df_fe['dosage_mg'] * df_fe['num_conds']

        fig_fe, ax_fe = plt.subplots(figsize=(8, 3))
        ax_fe.hist(df_fe['dosage_x_num_conds'], bins=40, color='#9b59b6', edgecolor='white', alpha=0.85)
        ax_fe.set_xlabel('dosage_x_num_conds'); ax_fe.set_ylabel('환자 수')
        ax_fe.set_title('상호작용 피처 분포', fontsize=12, fontweight='bold')
        ax_fe.spines['top'].set_visible(False); ax_fe.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_fe)

        st.subheader("최종 피처 목록")
        features = (
            [('age', '수치형', '나이'),
             ('dosage_mg', '수치형', '복용량 (mg)'),
             ('dosage_x_num_conds', '수치형 (파생)', '복용량 × 기저질환 수')] +
            [(f'cond_{c.replace(" ","_")}', 'Multi-Hot (이진)', f'기저질환: {CONDITION_KO.get(c,c)}') for c in COND_LIST] +
            [(f'drug_{d}', 'Multi-Hot (이진)', f'약물: {DRUG_KO.get(d,d)}') for d in DRUGS] +
            [('gender_*', 'One-Hot', '성별'),
             ('country_*', 'One-Hot', '국가'),
             ('smoker_*', 'One-Hot', '흡연 여부'),
             ('alcohol_use_*', 'One-Hot', '음주 여부')]
        )
        feat_df = pd.DataFrame(features, columns=['피처명', '인코딩 방식', '설명'])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── 탭4: 클래스 불균형 처리 ──────────────────────────────
    with tab4:
        st.subheader("원본 클래스 분포")
        orig_counts = df_raw['severity'].value_counts().reindex(['Mild', 'Moderate', 'Severe'])

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                orig_counts.rename_axis('중증도').reset_index(name='건수').assign(
                    비율=lambda x: (x['건수'] / x['건수'].sum() * 100).round(1).astype(str) + '%'
                ),
                use_container_width=True, hide_index=True
            )
        with col2:
            fig_o, ax_o = plt.subplots(figsize=(5, 3))
            ax_o.bar(orig_counts.index, orig_counts.values,
                     color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='white', width=0.5)
            ax_o.set_title('원본 클래스 분포', fontsize=11, fontweight='bold')
            ax_o.set_ylabel('환자 수')
            ax_o.spines['top'].set_visible(False); ax_o.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_o)

        st.subheader("해결 전략 비교")
        strategy_df = pd.DataFrame({
            '전략': ['클래스 가중치 조정', 'SMOTE 오버샘플링'],
            '방법': [
                'class_weight="balanced" — 소수 클래스에 높은 가중치 부여',
                'SMOTE(k_neighbors=5) — 소수 클래스 합성 샘플 생성'
            ],
            '장점': ['원본 데이터 그대로 유지', '클래스 수를 실제로 균등화'],
            '단점': ['극심한 불균형엔 한계', '합성 샘플이 노이즈 유발 가능'],
        })
        st.dataframe(strategy_df, use_container_width=True, hide_index=True)

        # 두 모델을 학습하고 macro F1 기준으로 자동 선택함을 안내
        st.info("두 모델을 모두 학습 후 **macro F1** 기준으로 더 나은 모델을 자동 선택합니다 (`train.py`).")

        st.subheader("평가 지표")
        metric_df = pd.DataFrame({
            '지표': ['Accuracy', 'Macro F1', 'Weighted F1', 'Confusion Matrix'],
            '설명': [
                '전체 정확도 — 클래스 불균형 시 Severe를 무시해도 높게 나올 수 있어 단독 사용 지양',
                '클래스별 F1 평균 (균등 가중치) — Severe 성능 저하를 민감하게 탐지',
                '클래스 크기 가중 F1 평균 — 전체 성능 요약',
                '클래스별 예측 오분류 패턴 확인 — Severe를 Mild로 잘못 예측하는 경우 포착',
            ]
        })
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
