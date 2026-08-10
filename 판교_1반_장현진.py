'''
====================================================================
 작성자 : 장현진
 작성일 : 2026.08.04
--------------------------------------------------------------------
 변경사항
 - 2026.08.04 / 상단 설명: 실습 4의 기능과 설치 방법을 정리했습니다.
 - 2026.08.04 / 함수 내부: 주요 변수·조건문 오른쪽에 처리 목적을 추가했습니다.
 - 2026.08.04 / 라이브러리 영역: requirements.txt 라이브러리 설명을 추가했습니다.
--------------------------------------------------------------------
 프로그램 설명
 - 정제 데이터로 시각화, 통계 검정, 머신러닝 파이프라인을 실행합니다.
--------------------------------------------------------------------
 실행 및 검사
 - practice3_analysis.py와 sales_100k.csv 파일을 같은 폴더에 배치합니다.
 - 가상환경 생성: python3 -m venv .venv
 - 가상환경 실행: source .venv/bin/activate
 - 라이브러리 설치: python -m pip install -r requirements.txt
 - 실행 명령어 : python 판교_1반_장현진.py
--------------------------------------------------------------------
 라이브러리 설명
 - pandas: 표 형식 데이터 처리 및 분석
 - matplotlib: 기본 그래프 생성
 - seaborn: 통계 그래프 시각화
 - scipy: 통계 검정 및 과학 계산
 - scikit-learn: 전처리·머신러닝 모델·평가
 - joblib: 학습한 모델 저장 및 불러오기
 - numpy: 수치 계산 및 배열 처리
 - plotly: 인터랙티브 그래프와 HTML 생성
====================================================================
'''

# ====================================================================
# 라이브러리 불러오기
# pip install -r requirements.txt                                  
from pathlib import Path
import sys                                                                # 운영체제 확인

import matplotlib                                                         # 그래프 출력 환경 설정
import pandas as pd                                                       # 표 형식 데이터 처리

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # 기본 그래프 생성
import seaborn as sns                                                     # 통계 그래프 시각화

from scipy import stats                                                   # 통계 검정

import joblib                                                             # 모델 저장 및 불러오기
import numpy as np                                                        # 수치 계산
from sklearn.compose import ColumnTransformer                             # 열별 전처리
from sklearn.impute import SimpleImputer                                  # 결측치 처리
from sklearn.linear_model import Ridge                                    # Ridge 회귀 모델
from sklearn.metrics import mean_squared_error                            # 오차 계산
from sklearn.model_selection import train_test_split                      # 학습·평가 데이터 분리
from sklearn.pipeline import Pipeline                                     # 전처리·모델 연결
from sklearn.preprocessing import OneHotEncoder, StandardScaler           # 범주형·수치형 전처리

import plotly.express as px                                               # 인터랙티브 그래프 생성
# ====================================================================

# ====================================================================
# 실습 3 모듈 연결
try:
    from practice3_analysis import aggregate_with_pandas, run_pandas_eda
except ImportError as error:
    raise SystemExit (f"모듈 가져오기 오류:{error}") from error

BASE_DIR = Path(__file__).resolve().parent                                # 현재 파일 폴더
OUTPUT_DIR = BASE_DIR / "outputs"                                         # 결과 저장 폴더
RANDOM_STATE = 42                                                         # 데이터 샘플링·분할 기준값
SIGNIFICANCE_LEVEL = 0.05                                                 # 통계적 유의수준
# ====================================================================

# ====================================================================
# 출력 폴더 및 그래프 스타일 설정
def prepare_output_directory() -> None:
                                                                          # 출력 폴더 생성
    OUTPUT_DIR.mkdir(exist_ok=True)                                       # 결과 폴더가 없으면 생성
    

def configure_plot_style() -> None:
                                                                          # 그래프 기본 스타일 및 macOS 한글 폰트 설정
    sns.set_theme(style="whitegrid")
    if sys.platform == "darwin":
        plt.rcParams["font.family"] = "AppleGothic"
        plt.rcParams["axes.unicode_minus"] = False
# ====================================================================

# ====================================================================
# EDA 대시보드 생성 및 저장
def create_eda_dashboard(clean_frame: pd.DataFrame) -> Path:
                                                                          # 그래프 스타일 설정 및 시각화용 데이터 샘플링
    configure_plot_style()                                                # 그래프 스타일 설정
    plot_frame = clean_frame.sample(
        n=min(100_000, len(clean_frame)),
        random_state = RANDOM_STATE
    )

                                                                          # 2×2 그래프 영역 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

                                                                          # 매출액 분포
    sns.histplot(data=plot_frame, x="amount", bins=40, kde=True, ax=axes[0,0])
    axes[0, 0].set_title("매출액 분포")
    axes[0, 0].set_xlabel("매출액")

                                                                          # 지역별 매출액 분포
    sns.boxplot(
        data=plot_frame,
        x="region",
        y="amount",
        showfliers=False,
        ax=axes[0,1],
    )
    axes[0, 1].set_title("지역별 매출액 분포")
    axes[0, 1].set_xlabel("지역")
    axes[0, 1].set_ylabel("매출액")

                                                                          # 월별 총매출 계산
    monthly = (
        clean_frame.assign(
            month=clean_frame["order_date"].dt.to_period("M").dt.to_timestamp()
        )
        .groupby("month", as_index=False)
        .agg(total=("amount", "sum"))
    )
    axes[1, 0].plot(
        monthly["month"], monthly["total"], marker="o", linewidth=2
    )
    axes[1, 0].set_title("월별 총매출 추이")
    axes[1, 0].set_xlabel("월")
    axes[1, 0].set_ylabel("총매출")
    axes[1, 0].tick_params(axis="x", rotation=45)

                                                                          # 수치형 변수 상관관계 계산
    numeric_columns = ["quantity", "unit_price", "customer_age", "amount"]
    correlation = clean_frame[numeric_columns].corr()
    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("수치형 변수 상관관계")

    fig.suptitle("매출 데이터 EDA 대시보드", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
                                                                          # 대시보드 이미지 저장
    output_path = OUTPUT_DIR / "eda_dashboard.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[1] EDA 대시보드 저장:{output_path.name}")
    return output_path
# ====================================================================

# ====================================================================
# 통계 검정
def run_statistical_tests(clean_frame: pd.DataFrame) -> None:
    """서울·부산 평균 검정과 지역·카테고리 독립성 검정을 수행합니다."""
                                                                          # 서울·부산 매출 데이터 추출 및 평균 검정
    seoul = clean_frame.loc[clean_frame["region"] == "서울", "amount"]
    busan = clean_frame.loc[clean_frame["region"] == "부산", "amount"]
    t_stat, t_pvalue = stats.ttest_ind(
        seoul, busan, equal_var=False, nan_policy="omit"
    )

    print("\n[2] 통계 검정")
    print(f"t-test: t={t_stat:.4f}, p={t_pvalue:.6f}")
                                                                          # t-test 결과 해석
    if t_pvalue < SIGNIFICANCE_LEVEL:                                     # t-test 유의성 판단
        print("해석: 서울과 부산의 평균 매출은 통계적으로 유의한 차이가 있습니다.")
    else:
        print("해석: 서울과 부산의 평균 매출은 통계적으로 유의한 차이가 없습니다.")

                                                                          # 지역·카테고리 교차표 생성 및 독립성 검정
    contingency = pd.crosstab(clean_frame["region"], clean_frame["category"])
    chi2, chi_pvalue, dof, _ = stats.chi2_contingency(contingency)
    print(f"카이제곱 검정: chi2={chi2:.4f}, 자유도={dof}, p={chi_pvalue:.6f}")
    if chi_pvalue < SIGNIFICANCE_LEVEL:                                   # 카이제곱 검정 유의성 판단
        print("해석: 지역과 카테고리는 서로 독립적이지 않습니다.")
    else:
        print("해석: 지역과 카테고리가 독립적이라는 가설을 기각할 수 없습니다.")
# ====================================================================

# ====================================================================
# sklearn 전처리·모델 학습·저장 및 재로딩
def train_and_save_pipeline(clean_frame: pd.DataFrame) -> Path:
    """전처리와 Ridge 모델을 하나의 Pipeline으로 학습하고 저장합니다."""
                                                                          # 학습 특성 및 대상 변수 설정
    numeric_features = ["quantity", "unit_price", "customer_age"]
    categorical_features = [
        "region",
        "category",
        "payment_method",
        "customer_gender",
    ]
    feature_columns = numeric_features + categorical_features

                                                                          # 결측치 제거 및 학습·평가 데이터 분리
    model_frame = clean_frame[feature_columns + ["amount"]].dropna().sample(
        n=min(200_000, len(clean_frame)), random_state=RANDOM_STATE
    )
    features = model_frame[feature_columns]
    target = model_frame["amount"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

                                                                          # 수치형·범주형 전처리 파이프라인 구성
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )

                                                                          # 모델 학습 및 평가
    model_pipeline.fit(x_train, y_train)
    predictions = model_pipeline.predict(x_test)
    r2 = model_pipeline.score(x_test, y_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

                                                                          # 모델 저장 및 재로딩 확인
    model_path = OUTPUT_DIR / "sales_ridge_pipeline.joblib"
    joblib.dump(model_pipeline, model_path)
    loaded_pipeline = joblib.load(model_path)
    reload_matches = np.allclose(
        predictions[:10], loaded_pipeline.predict(x_test.head(10))
    )

    print("\n[3] sklearn Pipeline")
    print(f"학습 행 수:{len(x_train):,}, 평가 행 수:{len(x_test):,}")
    print(f"R²:{r2:.4f}, RMSE:{rmse:,.2f}")
    print(f"모델 저장:{model_path.name}")
    print(f"재로딩 예측 일치:{reload_matches}")
    return model_path
# ====================================================================

# ====================================================================
# Plotly 인터랙티브 차트 생성 및 HTML 저장
def create_plotly_chart(summary: pd.DataFrame) -> Path:
    """지역·카테고리별 총매출 막대 차트를 HTML로 저장합니다."""
                                                                          # 지역·카테고리별 총매출 막대 차트 생성
    fig = px.bar(
        summary,
        x="region",
        y="total",
        color="category",
        barmode="group",
        hover_data={"mean": ":,.0f", "count": ":,"},
        title="지역·카테고리별 총매출",
        labels={"region": "지역", "total": "총매출", "category": "카테고리"},
    )
    fig.update_layout(template="plotly_white")
                                                                          # 인터랙티브 HTML 저장
    output_path = OUTPUT_DIR / "regional_category_sales.html"
    fig.write_html(output_path, include_plotlyjs=True)
    print(f"\n[4] Plotly 차트 저장:{output_path.name}")
    return output_path
# ====================================================================

# ====================================================================
# 전체 실습 단계 실행
def main() -> None:
                                                                          # 실습 실행에 필요한 폴더와 정제 데이터 준비
    prepare_output_directory()
    clean_frame, _, _ = run_pandas_eda()                                  # 실습 3 정제 데이터 준비
    summary = aggregate_with_pandas(clean_frame)                          # Pandas 집계 결과 생성
                                                                          # 분석 및 시각화 단계 실행
    aggregate_with_pandas(clean_frame)                                    # 집계 결과 출력
    create_eda_dashboard(clean_frame)                                     # EDA 대시보드 생성
    run_statistical_tests(clean_frame)                                    # 통계 검정 실행
    train_and_save_pipeline(clean_frame)                                  # 머신러닝 파이프라인 실행
    create_plotly_chart(summary)                                          # Plotly HTML 차트 생성
    print("\n실습 4 완료: outputs 폴더를 확인합니다.")
# ====================================================================

# ====================================================================
# 프로그램 실행 중 오류 확인
if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"실행 오류:{error}")
# ====================================================================
