"""한국농수산식품유통공사(aT) 농축수산물 표준코드(katCode) 조회 스크립트.

공영도매시장 표준 품목(소분류) 코드를 조회하고 CSV로 저장합니다.
- 대분류: 06 (과실류)
- 중분류: 14 (감귤), 15 (만감: 한라봉, 천혜향, 레드향 등)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

# ---------------------------------------------------------------------
# API 설정
# ---------------------------------------------------------------------
# URL에 중복 슬래시(//)가 포함된 경우 공공데이터포털에서 400 오류가 발생하므로 정규화합니다.
DEFAULT_URL = "https://apis.data.go.kr/B552845/katCode/goods"

# .env의 DATA_GO_KR_SERVICE_KEY를 기본으로 사용하고, 없으면 전달받은 기본 키 사용
SERVICE_KEY = os.getenv(
    "DATA_GO_KR_SERVICE_KEY",
    "a212bee4ca445f500361d08864230ae94ce6d4641b092ed95788f6cd3b29cf2e",
).strip()

OUTPUT_DIR = Path("data")


def clean_url(url: str) -> str:
    """URL의 중복 슬래시(//)를 단일 슬래시(/)로 정규화합니다 (프로토콜 https:// 제외)."""
    return re.sub(r"(?<!:)//+", "/", url.strip())


def normalize_code(code: str | int | None, length: int = 2) -> str | None:
    """코드 값이 1자리인 경우 2자리 표준(0-padded)으로 변환합니다 (예: 6 -> '06')."""
    if code is None:
        return None
    val = str(code).strip()
    return val.zfill(length) if val.isdigit() and len(val) < length else val


def fetch_standard_goods_codes(
    url: str = DEFAULT_URL,
    service_key: str = SERVICE_KEY,
    lclsf_cd: str | int = "06",
    mclsf_cd: str | int | None = "14",
    selectable: str = "gds_sclsf_cd",
    num_of_rows: int = 100,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """aT 공영도매시장 표준 품목(소분류) 코드 목록을 API로 조회합니다.

    Parameters
    ----------
    url : str
        표준코드 조회 API 엔드포인트 URL
    service_key : str
        공공데이터포털 인증키
    lclsf_cd : str | int
        대분류 코드 (기본값 '06': 과실류, '6' 입력 시 '06'으로 자동 변환)
    mclsf_cd : str | int | None
        중분류 코드 (기본값 '14': 감귤, '15': 만감류, None일 경우 전체)
    selectable : str
        조회 필드 지정 (기본값 'gds_sclsf_cd')
    num_of_rows : int
        한 페이지당 조회 건수 (기본값 100)
    timeout : int
        요청 타임아웃(초)

    Returns
    -------
    list[dict[str, Any]]
        표준코드 목록 딕셔너리 리스트
    """
    clean_endpoint = clean_url(url)
    norm_lclsf = normalize_code(lclsf_cd, 2)
    norm_mclsf = normalize_code(mclsf_cd, 2) if mclsf_cd is not None else None

    params: dict[str, Any] = {
        "serviceKey": service_key,
        "returnType": "json",
        "numOfRows": num_of_rows,
        "pageNo": 1,
    }

    if norm_lclsf:
        params["cond[gds_lclsf_cd::EQ]"] = norm_lclsf
    if norm_mclsf:
        params["cond[gds_mclsf_cd::EQ]"] = norm_mclsf
    if selectable:
        params["selectable"] = selectable

    all_items: list[dict[str, Any]] = []
    page_no = 1

    print(f"[INFO] API 요청 시작: {clean_endpoint}")
    print(f"       대분류(gds_lclsf_cd): {norm_lclsf}, 중분류(gds_mclsf_cd): {norm_mclsf}")

    while True:
        params["pageNo"] = page_no
        try:
            response = requests.get(clean_endpoint, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] API 요청 실패 (page {page_no}): {e}", file=sys.stderr)
            break
        except Exception as e:
            print(f"[ERROR] 응답 파싱 실패 (page {page_no}): {e}", file=sys.stderr)
            break

        body = data.get("response", {}).get("body", {})
        total_count = body.get("totalCount", 0)
        items_data = body.get("items", {}).get("item", [])

        if isinstance(items_data, dict):
            items_data = [items_data]

        if not items_data:
            break

        all_items.extend(items_data)
        print(f"       {page_no}페이지 수신 완료: {len(items_data)}건 (누적 {len(all_items)} / 전체 {total_count})")

        if len(all_items) >= total_count:
            break

        page_no += 1

    return all_items


def main() -> None:
    # 1. 감귤류 (대분류: 06 과실류, 중분류: 14 감귤) 조회
    print("=" * 60)
    print("1. [감귤] 표준 품목(소분류) 코드 조회 (대분류=06, 중분류=14)")
    print("=" * 60)

    # 사용자가 전달한 조건 (URL: katCode//goods, cond: 6, 14, selectable: gds_sclsf_cd)
    citrus_items = fetch_standard_goods_codes(
        url="https://apis.data.go.kr/B552845/katCode//goods",
        service_key=SERVICE_KEY,
        lclsf_cd="6",  # 6 입력 시 06으로 자동 변환 처리
        mclsf_cd="14",
        selectable="gds_sclsf_cd",
    )

    if citrus_items:
        df_citrus = pd.DataFrame(citrus_items)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_citrus_path = OUTPUT_DIR / "standard_goods_code_citrus_14.csv"
        df_citrus.to_csv(csv_citrus_path, index=False, encoding="utf-8-sig")

        print("\n[감귤(15) 소분류 코드 목록]")
        print(df_citrus[["gds_sclsf_cd", "gds_sclsf_nm", "gds_mclsf_nm", "gds_lclsf_nm"]].to_string(index=False))
        print(f"\n-> 총 {len(df_citrus)}건 저장 완료: {csv_citrus_path}")

    # 2. 만감류 (대분류: 06 과실류, 중분류: 15 만감 - 한라봉, 천혜향, 레드향 등) 추가 조회
    print("\n" + "=" * 60)
    print("2. [참고] [만감] 표준 품목 코드 조회 (대분류=06, 중분류=15)")
    print("   * 한라봉(04), 천혜향(07), 황금향(08), 레드향(09), 카라향(11) 등")
    print("=" * 60)

    tangor_items = fetch_standard_goods_codes(
        url=DEFAULT_URL,
        service_key=SERVICE_KEY,
        lclsf_cd="06",
        mclsf_cd="15",
        selectable="gds_sclsf_cd",
    )

    if tangor_items:
        df_tangor = pd.DataFrame(tangor_items)
        csv_tangor_path = OUTPUT_DIR / "standard_goods_code_tangor_15.csv"
        df_tangor.to_csv(csv_tangor_path, index=False, encoding="utf-8-sig")

        print("\n[만감(15) 주요 소분류 코드]")
        focus_codes = ["04", "07", "08", "09", "10", "11", "27"]
        df_focus = df_tangor[df_tangor["gds_sclsf_cd"].isin(focus_codes)]
        print(df_focus[["gds_sclsf_cd", "gds_sclsf_nm", "gds_mclsf_nm"]].to_string(index=False))
        print(f"\n-> 총 {len(df_tangor)}건 저장 완료: {csv_tangor_path}")


if __name__ == "__main__":
    main()
