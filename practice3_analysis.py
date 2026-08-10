'''
====================================================================
 작성자 : 장현진
 작성일 : 2026.08.03
--------------------------------------------------------------------
 변경사항
 - Pandas 데이터 탐색 기능 추가
 - IQR 기반 amount 이상치 제거 기능 추가
 - region, category 결측치 제거 기능 추가
--------------------------------------------------------------------
 프로그램 설명
 - sales_100k.csv의 데이터 구조와 결측치를 확인합니다.
 - amount 컬럼의 IQR 정상 범위를 계산하고 이상치를 제거합니다.
 - 집계에 필요한 region, category 결측치를 제거합니다.

 실행 및 검사
 - sales_100k.csv 파일을 Python 파일과 같은 폴더에 배치합니다.
 - 실행 명령어 : python practice03.py or 해당 파일에서 직접 실행

====================================================================
'''

# ====================================================================
# 라이브러리 불러오기
# pip install -r requirements.txt
from pathlib import Path
import pandas as pd
import polars as pl
import duckdb
# ====================================================================

# ====================================================================
# CSV파일 경로 설정
DATA_PATH = Path(__file__).with_name("sales_100k.csv")

def ensure_data_file() -> None:                                         # CSV파일 존재 여부 확인 함수선언
    """입력 CSV파일의 존재 여부를 확인합니다."""
    if not DATA_PATH.exists():                                           # CSV파일이 없을 경우 예외처리
        raise FileNotFoundError(
            f"{DATA_PATH.name}이 없습니다. 해당 파일 위치를 확인해주세요."
        )
# ====================================================================

# ====================================================================
# 데이터 탐색, 데이터 IQR 기반 전처리
def run_pandas_eda() -> tuple[pd.DataFrame, float, float]:              # 데이터 탐색 및 전처리 전체 함수선언
    """데이터를 탐색하고 IQR 기준으로 amount 이상치를 제거합니다."""
    ensure_data_file()                                                  # CSV파일 존재 여부 우선 확인
    try:                                                                # order_date를 날짜형으로 변환하여 CSV파일 불러오기
        frame = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise RuntimeError(f"CSV 파일을 읽지 못했습니다:{error}") from error

    required_columns = {"region", "category", "amount"}                 # 데이터 처리에 필요한 필수 컬럼 설정
    missing_columns = required_columns.difference(frame.columns)        # CSV파일에 없는 필수 컬럼 확인

    if missing_columns:                                                 # 필수 컬럼이 없을 경우 예외처리
        raise ValueError(f"열이 없습니다:{sorted(missing_columns)}")
# 데이터 기본 정보 및 결측치 출력
    print(f"[1] 데이터 기본 정보")                                  
    print(f"크기:{frame.shape}")
    frame.info()
    print(f"결측치 수")
    print(frame.isnull().sum())

    amount = pd.to_numeric(frame["amount"], errors="coerce")           # amount를 숫자형으로 변환, 변환 불가 값은 결측치 처리

    q1 = amount.quantile(0.25)                                          # amount의 1사분위수 계산
    q3 = amount.quantile(0.75)                                          # amount의 3사분위수 계산
    iqr = q3 - q1                                                       # 3사분위수와 1사분위수의 차이 계산
    lower = q1 - 1.5 * iqr                                              # IQR 정상 범위의 하한 설정
    upper = q3 + 1.5 * iqr                                              # IQR 정상 범위의 상한 설정

    iqr_frame = frame[amount.between(lower, upper)].copy()               # IQR 정상 범위에 포함되는 행만 선택
    clean_frame = iqr_frame.dropna(subset=["region", "category"]).copy() # 집계 기준 컬럼의 결측 행 제거
    clean_frame["amount"] = pd.to_numeric(clean_frame["amount"])         # 정제된 amount 컬럼을 숫자형으로 저장

    print(f"\nIQR 정상 범위:{lower:,.2f} ~{upper:,.2f}")
    print(f"IQR 적용 전 행 수:{len(frame):,}")
    print(f"IQR 적용 후 행 수:{len(iqr_frame):,}")
    print(f"집계 키 결측 제거 후 행 수:{len(clean_frame):,}")
    print(f"총 제외 행 수:{len(frame) - len(clean_frame):,}")
    return clean_frame, float(lower), float(upper)                       # 정제 데이터와 IQR 하한 및 상한 반환

def aggregate_with_pandas(clean_frame: pd.DataFrame) -> pd.DataFrame:
    # ====================================================================
    # Pandas 그룹별 집계
    result = (
        clean_frame.groupby(["region", "category"], as_index=False)   # 지역·상품분류별 그룹화
        .agg(
            total=("amount", "sum"),                                  # amount 총합 계산
            mean=("amount", "mean"),                                  # amount 평균 계산
            count=("amount", "count"),                                # amount 데이터 개수 계산
        )
        .sort_values("total", ascending=False)                        # 총매출 기준 내림차순 정렬
        .reset_index(drop=True)                                       # 인덱스 재설정
    )
    print("\n[2] Pandas 집계 결과 상위 5행")                              # 집계 결과 제목 출력
    print(result.head().to_string(index=False))                       # 집계 결과 상위 5행 출력
    return result                                                     # Pandas 집계 결과 반환
# ====================================================================

def build_polars_query(lower: float, upper: float) -> pl.LazyFrame:
    # ====================================================================
    # Polars Lazy API 집계 계획 작성
    return (
        pl.scan_csv(DATA_PATH)                                         # CSV파일을 LazyFrame으로 읽기
        .filter(
            pl.col("amount").is_between(lower, upper, closed="both")   # IQR 정상 범위 적용
            & pl.col("region").is_not_null()                           # region 결측치 제거
            & pl.col("category").is_not_null()                         # category 결측치 제거
        )
        .group_by(["region", "category"])                              # 지역·상품분류별 그룹화
        .agg(
            pl.col("amount").sum().alias("total"),                     # amount 총합 계산
            pl.col("amount").mean().alias("mean"),                     # amount 평균 계산
            pl.col("amount").count().alias("count"),                   # amount 데이터 개수 계산
        )
        .sort("total", descending=True)                                # 총매출 내림차순 정렬
    )

def aggregate_with_polars(lower: float, upper: float) -> pl.DataFrame:
    # ====================================================================
    # Polars Lazy API 실행 및 결과 출력
    result = build_polars_query(lower, upper).collect()                 # Lazy 실행 계획 실행
    print("\n[3] Polars 집계 결과 상위 5행")                                # 결과 제목 출력
    print(result.head())                                                # 집계 결과 상위 5행 출력
    return result                                                       # Polars 결과 반환
# ====================================================================

def build_duckdb_query(lower: float, upper: float) -> str:
    # ====================================================================
    # DuckDB SQL 집계문 작성
    safe_path = DATA_PATH.as_posix().replace("'", "''")                 # SQL용 CSV 경로 설정
    return f"""
        SELECT
            region,
            category,
            SUM(amount) AS total,
            AVG(amount) AS mean,
            COUNT(amount) AS count
        FROM read_csv('{safe_path}', header = true)
        WHERE amount BETWEEN {lower} AND {upper}                         -- IQR 정상 범위 적용
          AND region IS NOT NULL                                         -- region 결측치 제거
          AND category IS NOT NULL                                       -- category 결측치 제거
        GROUP BY region, category                                        -- 지역·상품분류별 그룹화
        ORDER BY total DESC                                              -- 총매출 내림차순 정렬
    """


def aggregate_with_duckdb(lower: float, upper: float) -> pd.DataFrame:
    # ====================================================================
    # DuckDB SQL 실행 및 결과 출력
    result = duckdb.sql(build_duckdb_query(lower, upper)).df()          # SQL 실행 후 DataFrame 변환
    print("\n[4] DuckDB 집계 결과 상위 5행")                                # 결과 제목 출력
    print(result.head().to_string(index=False))                         # 집계 결과 상위 5행 출력
    return result                                                       # DuckDB 결과 반환
# ====================================================================

def main() -> None:
    # ====================================================================
    # 전체 실습 단계 실행
    clean_frame, lower, upper = run_pandas_eda()                        # Pandas EDA 및 전처리 실행
    aggregate_with_pandas(clean_frame)                                  # Pandas 집계 실행
    aggregate_with_polars(lower, upper)                                 # Polars Lazy 집계 실행
    aggregate_with_duckdb(lower, upper)                                 # DuckDB SQL 집계 실행
    # ====================================================================

# ====================================================================
# 프로그램 실행 중 오류 확인
if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, duckdb.Error) as error:
        print(f"실행 오류:{error}")
# ====================================================================