from __future__ import annotations

import calendar
import csv
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import requests
from dotenv import load_dotenv


load_dotenv()


# ---------------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------------

BASE_URL = "https://apis.data.go.kr/B552845/katSale/trades"  # 공영도매시장 정산정보 조회 서비스

# 일부 공공데이터 API는 서비스 기본 URL 뒤에 상세 오퍼레이션명이 필요합니다.
#
# 공공데이터포털 상세 페이지에 표시된 실제 요청 URL에 operation이 있다면
# 아래 환경변수 API_OPERATION으로 지정합니다.
#
# 예:
# API_OPERATION=/saleInfo
# API_OPERATION=/trades
#
# 상세 경로가 필요하지 않은 경우에는 빈 문자열로 둡니다.
API_OPERATION = os.getenv("API_OPERATION", "").strip()

SERVICE_KEY = os.getenv("DATA_GO_KR_SERVICE_KEY", "").strip()

# 조회 기간
START_DATE = os.getenv("START_DATE", "2026-06-01").strip()
END_DATE = os.getenv("END_DATE", "2026-11-11").strip()

# 품목 조건
LARGE_CATEGORY_CODE = "06"  # 과실류
MIDDLE_CATEGORY_CODE = "14"  # 감귤

# API 페이지 설정
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "1000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
REQUEST_RETRY_COUNT = int(os.getenv("REQUEST_RETRY_COUNT", "5"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.2"))

# 실행 중 오류가 발생해도 다음 월의 조회를 계속할지 여부
CONTINUE_ON_ERROR = (
    os.getenv("CONTINUE_ON_ERROR", "true").lower()
    in ("1", "true", "yes", "y")
)

# 출력 폴더
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output"))


# ---------------------------------------------------------------------
# 첨부파일에서 확인된 공영도매시장 목록
# ---------------------------------------------------------------------

MARKETS: list[dict[str, str]] = [
    #{"whsl_mrkt_cd": "110001", "whsl_mrkt_nm": "서울가락"},
    #{"whsl_mrkt_cd": "110008", "whsl_mrkt_nm": "서울강서"},
    #{"whsl_mrkt_cd": "210001", "whsl_mrkt_nm": "부산엄궁"},
    #{"whsl_mrkt_cd": "210005", "whsl_mrkt_nm": "부산국제수산"}
    #{"whsl_mrkt_cd": "210009", "whsl_mrkt_nm": "부산반여"}
    #{"whsl_mrkt_cd": "220001", "whsl_mrkt_nm": "대구북부"},
    #{"whsl_mrkt_cd": "230001", "whsl_mrkt_nm": "인천남촌"},
    #{"whsl_mrkt_cd": "230003", "whsl_mrkt_nm": "인천삼산"}
     {"whsl_mrkt_cd": "240001", "whsl_mrkt_nm": "광주각화"},
     {"whsl_mrkt_cd": "240004", "whsl_mrkt_nm": "광주서부"},
     {"whsl_mrkt_cd": "250001", "whsl_mrkt_nm": "대전오정"},
     {"whsl_mrkt_cd": "250003", "whsl_mrkt_nm": "대전노은"},
     {"whsl_mrkt_cd": "310101", "whsl_mrkt_nm": "수원"}
    # {"whsl_mrkt_cd": "310401", "whsl_mrkt_nm": "안양"},
    # {"whsl_mrkt_cd": "310901", "whsl_mrkt_nm": "안산"},
    # {"whsl_mrkt_cd": "311201", "whsl_mrkt_nm": "구리"},
    # {"whsl_mrkt_cd": "320101", "whsl_mrkt_nm": "춘천"},
    # {"whsl_mrkt_cd": "320201", "whsl_mrkt_nm": "원주"},
    # {"whsl_mrkt_cd": "320301", "whsl_mrkt_nm": "강릉"},
    # {"whsl_mrkt_cd": "330101", "whsl_mrkt_nm": "청주"},
    # {"whsl_mrkt_cd": "330201", "whsl_mrkt_nm": "충주"},
    # {"whsl_mrkt_cd": "340101", "whsl_mrkt_nm": "천안"},
    # {"whsl_mrkt_cd": "350101", "whsl_mrkt_nm": "전주"},
    # {"whsl_mrkt_cd": "350301", "whsl_mrkt_nm": "익산"},
    # {"whsl_mrkt_cd": "350402", "whsl_mrkt_nm": "정읍"},
    # {"whsl_mrkt_cd": "360301", "whsl_mrkt_nm": "순천"},
    # {"whsl_mrkt_cd": "370101", "whsl_mrkt_nm": "포항"},
    # {"whsl_mrkt_cd": "370401", "whsl_mrkt_nm": "안동"},
    # {"whsl_mrkt_cd": "371501", "whsl_mrkt_nm": "구미"},
    # {"whsl_mrkt_cd": "380101", "whsl_mrkt_nm": "창원팔용"},
    # {"whsl_mrkt_cd": "380201", "whsl_mrkt_nm": "울산"},
    # {"whsl_mrkt_cd": "380303", "whsl_mrkt_nm": "창원내서"},
    # {"whsl_mrkt_cd": "380401", "whsl_mrkt_nm": "진주"},
]


# ---------------------------------------------------------------------
# 예외 클래스
# ---------------------------------------------------------------------

class KatSaleApiError(Exception):
    """공영도매시장 정산정보 API 오류"""


# ---------------------------------------------------------------------
# 공통 유틸리티
# ---------------------------------------------------------------------

def parse_date(value: str) -> date:
    """YYYY-MM-DD 형식 문자열을 date 객체로 변환합니다."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"날짜 형식이 올바르지 않습니다: {value}. "
            "YYYY-MM-DD 형식을 사용하세요."
        ) from exc


def month_ranges(
    start: date,
    end: date,
) -> Iterator[tuple[date, date]]:
    """
    시작일~종료일을 월 단위 범위로 나눕니다.

    예:
    2020-01-01 ~ 2020-03-15
    -> 2020-01-01 ~ 2020-01-31
    -> 2020-02-01 ~ 2020-02-29
    -> 2020-03-01 ~ 2020-03-15
    """
    current = date(start.year, start.month, 1)

    while current <= end:
        last_day = calendar.monthrange(
            current.year,
            current.month,
        )[1]

        range_start = max(start, current)
        range_end = min(
            end,
            date(current.year, current.month, last_day),
        )

        yield range_start, range_end

        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(
                current.year,
                current.month + 1,
                1,
            )


def normalize_items(items: Any) -> list[dict[str, Any]]:
    """
    API 응답의 item이 다음처럼 다양하게 반환될 수 있어 통일합니다.

    - item: [...]
    - item: {...}
    - item: null
    """
    if items is None:
        return []

    if isinstance(items, dict):
        return [items]

    if isinstance(items, list):
        return [
            item for item in items
            if isinstance(item, dict)
        ]

    return []


def get_api_url() -> str:
    """기본 URL과 상세 오퍼레이션을 결합합니다."""
    if not API_OPERATION:
        return BASE_URL

    if API_OPERATION.startswith("/"):
        return BASE_URL.rstrip("/") + API_OPERATION

    return BASE_URL.rstrip("/") + "/" + API_OPERATION


def make_output_filename(
    range_start: date,
    range_end: date,
) -> str:
    return (
        f"kat_sale_citrus3_"
        f"{range_start.strftime('%Y%m%d')}_"
        f"{range_end.strftime('%Y%m%d')}.csv"
    )


# ---------------------------------------------------------------------
# API 클라이언트
# ---------------------------------------------------------------------

class KatSaleClient:
    def __init__(
        self,
        service_key: str,
        timeout: int = 60,
        retry_count: int = 5,
    ) -> None:
        if not service_key:
            raise ValueError(
                "서비스키가 없습니다. "
                ".env 파일의 DATA_GO_KR_SERVICE_KEY를 설정하세요."
            )

        self.service_key = service_key
        self.timeout = timeout
        self.retry_count = retry_count
        self.session = requests.Session()

    def request_page(
        self,
        *,
        page_no: int,
        num_of_rows: int,
        target_date: date | str | None = None,
        market_code: str,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        정산정보 API의 한 페이지를 조회합니다.

        실제 API 규격:
        "cond[trd_clcln_ymd::EQ]"를 통해 특정 단일 일자(하루)의 데이터를 조회합니다.
        """
        query_date = target_date if target_date is not None else start_date
        if query_date is None:
            raise ValueError("조회 일자(target_date 또는 start_date)가 지정되지 않았습니다.")

        ymd_str = (
            query_date.isoformat()
            if isinstance(query_date, (date, datetime))
            else str(query_date).strip()
        )

        url = get_api_url()

        params: dict[str, Any] = {
            "serviceKey": self.service_key,
            "returnType": "json",
            "pageNo": page_no,
            "numOfRows": num_of_rows,

            # 시장
            "cond[whsl_mrkt_cd::EQ]": market_code,

            # 품목: 과실류(06), 감귤(14)
            "cond[gds_lclsf_cd::EQ]": LARGE_CATEGORY_CODE,
            "cond[gds_mclsf_cd::EQ]": MIDDLE_CATEGORY_CODE,

            # 정산일 (단일 일자 EQ 조건)
            "cond[trd_clcln_ymd::EQ]": ymd_str,
        }

        last_error: Exception | None = None

        for attempt in range(1, self.retry_count + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    raise KatSaleApiError(
                        f"HTTP {response.status_code}: "
                        f"{response.text[:1000]}"
                    )

                # JSON 응답을 우선 처리합니다.
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise KatSaleApiError(
                        "JSON 응답이 아닙니다.\n"
                        f"요청 URL: {response.url}\n"
                        f"응답 내용: {response.text[:1000]}"
                    ) from exc

                return self.parse_response(payload)

            except (
                requests.RequestException,
                KatSaleApiError,
            ) as exc:
                last_error = exc

                if attempt == self.retry_count:
                    break

                sleep_seconds = min(2 ** (attempt - 1), 20)

                print(
                    f"  요청 실패: {exc}\n"
                    f"  {sleep_seconds}초 후 재시도 "
                    f"({attempt}/{self.retry_count})"
                )

                time.sleep(sleep_seconds)

        raise KatSaleApiError(
            f"API 요청을 최종 실패했습니다: {last_error}"
        )

    @staticmethod
    def parse_response(
        payload: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        """
        공공데이터포털 표준 응답에서 item과 totalCount를 추출합니다.
        """
        if not isinstance(payload, dict):
            raise KatSaleApiError(
                "API 응답이 JSON 객체 형식이 아닙니다."
            )

        response_data = payload.get("response", payload)

        header = response_data.get("header", {})
        result_code = str(header.get("resultCode", ""))
        result_message = header.get("resultMsg", "")

        # 성공 코드는 API에 따라 00 또는 0으로 올 수 있습니다.
        if result_code not in ("", "00", "0"):
            raise KatSaleApiError(
                f"API 오류 [{result_code}] {result_message}\n"
                f"원본 응답: {json.dumps(payload, ensure_ascii=False)[:2000]}"
            )

        body = response_data.get("body", {})

        total_count = int(
            body.get("totalCount", 0) or 0
        )

        raw_items = (
            body
            .get("items", {})
            .get("item", [])
        )

        items = normalize_items(raw_items)

        return items, total_count

    def fetch_market_date(
        self,
        *,
        market_code: str,
        market_name: str,
        target_date: date,
    ) -> list[dict[str, Any]]:
        """
        한 시장의 특정 하루(target_date)에 대한 전체 페이지를 조회합니다.
        """
        all_items: list[dict[str, Any]] = []
        page_no = 1
        ymd_str = target_date.isoformat()

        while True:
            items, total_count = self.request_page(
                page_no=page_no,
                num_of_rows=PAGE_SIZE,
                target_date=target_date,
                market_code=market_code,
            )

            # 시장 정보와 조회 일자 조건을 결과에 추가합니다.
            for item in items:
                item.setdefault("query_whsl_mrkt_cd", market_code)
                item.setdefault("query_whsl_mrkt_nm", market_name)
                item.setdefault("query_trd_date", ymd_str)
                item.setdefault("query_start_date", ymd_str)
                item.setdefault("query_end_date", ymd_str)

            all_items.extend(items)

            if not items:
                break

            if total_count > 0 and len(all_items) >= total_count:
                break

            # API가 totalCount를 제공하지 않는 경우에도 마지막 페이지를 판단할 수 있도록 처리
            if len(items) < PAGE_SIZE:
                break

            page_no += 1
            time.sleep(REQUEST_DELAY)

        return all_items

    def fetch_market_range(
        self,
        *,
        market_code: str,
        market_name: str,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        """
        한 시장의 지정된 기간에 대해 일자별(cond[trd_clcln_ymd::EQ])로 순회하며 전체 데이터를 수집합니다.
        """
        all_items: list[dict[str, Any]] = []
        total_days = (end_date - start_date).days + 1

        print(
            f"    {market_name}({market_code}) "
            f"{start_date} ~ {end_date} (총 {total_days}일 일별 순회 조회)"
        )

        current_date = start_date
        day_idx = 0
        while current_date <= end_date:
            day_idx += 1
            day_items = self.fetch_market_date(
                market_code=market_code,
                market_name=market_name,
                target_date=current_date,
            )

            # 검색 구간 메타데이터 갱신
            for item in day_items:
                item["query_start_date"] = start_date.isoformat()
                item["query_end_date"] = end_date.isoformat()

            all_items.extend(day_items)

            if len(day_items) > 0:
                print(
                    f"      [{day_idx}/{total_days}] {current_date}: "
                    f"{len(day_items):,}건 수신 (누적 {len(all_items):,}건)"
                )

            current_date += timedelta(days=1)

        return all_items


# ---------------------------------------------------------------------
# CSV 저장
# ---------------------------------------------------------------------

def write_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    JSON 응답의 서로 다른 필드를 모두 열로 만들어 CSV로 저장합니다.

    한 행마다 필드가 다를 수 있으므로 전체 행에서 열 이름을 수집합니다.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        # 데이터가 없는 경우에도 빈 CSV 파일을 생성합니다.
        output_path.write_text(
            "",
            encoding="utf-8-sig",
        )
        return

    fieldnames: list[str] = []
    fieldname_set: set[str] = set()

    for row in rows:
        for key in row.keys():
            key = str(key)

            if key not in fieldname_set:
                fieldname_set.add(key)
                fieldnames.append(key)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            normalized_row: dict[str, Any] = {}

            for fieldname in fieldnames:
                value = row.get(fieldname, "")

                # 중첩 객체나 배열이 있으면 JSON 문자열로 저장
                if isinstance(value, (dict, list)):
                    value = json.dumps(
                        value,
                        ensure_ascii=False,
                    )

                normalized_row[fieldname] = value

            writer.writerow(normalized_row)

    print(
        f"CSV 저장 완료: {output_path} "
        f"({len(rows):,}건)"
    )


def write_error_log(
    errors: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    if not errors:
        return

    error_path = output_dir / "errors.json"

    error_path.write_text(
        json.dumps(
            errors,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"오류 로그 저장 완료: {error_path}"
    )


# ---------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------

def main() -> int:
    if not SERVICE_KEY:
        print(
            "오류: DATA_GO_KR_SERVICE_KEY가 설정되지 않았습니다.\n"
            ".env 파일을 확인하세요."
        )
        return 1

    start_date = parse_date(START_DATE)
    end_date = parse_date(END_DATE)

    if start_date > end_date:
        print(
            f"오류: 시작일({start_date})이 "
            f"종료일({end_date})보다 늦습니다."
        )
        return 1

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = KatSaleClient(
        service_key=SERVICE_KEY,
        timeout=REQUEST_TIMEOUT,
        retry_count=REQUEST_RETRY_COUNT,
    )

    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    ranges = list(
        month_ranges(
            start_date,
            end_date,
        )
    )

    print("공영도매시장 정산정보 다운로드를 시작합니다.")
    print(f"- API URL: {get_api_url()}")
    print(f"- 조회 기간: {start_date} ~ {end_date}")
    print(f"- 대분류: {LARGE_CATEGORY_CODE} 과실류")
    print(f"- 중분류: {MIDDLE_CATEGORY_CODE} 감귤")
    print(f"- 시장 수: {len(MARKETS)}개")
    print(f"- 월별 구간 수: {len(ranges)}개")
    print()

    for range_index, (range_start, range_end) in enumerate(
        ranges,
        start=1,
    ):
        print(
            f"[{range_index}/{len(ranges)}] "
            f"{range_start} ~ {range_end}"
        )

        range_rows: list[dict[str, Any]] = []

        for market in MARKETS:
            market_code = market["whsl_mrkt_cd"]
            market_name = market["whsl_mrkt_nm"]

            try:
                rows = client.fetch_market_range(
                    market_code=market_code,
                    market_name=market_name,
                    start_date=range_start,
                    end_date=range_end,
                )

                range_rows.extend(rows)
                all_rows.extend(rows)

            except Exception as exc:
                error = {
                    "market_code": market_code,
                    "market_name": market_name,
                    "start_date": range_start.isoformat(),
                    "end_date": range_end.isoformat(),
                    "error": str(exc),
                }

                errors.append(error)

                print(
                    f"  오류: {market_name}({market_code}) - {exc}"
                )

                if not CONTINUE_ON_ERROR:
                    raise

        # 월별 파일 저장
        monthly_filename = make_output_filename(
            range_start,
            range_end,
        )

        write_csv(
            range_rows,
            OUTPUT_DIR / monthly_filename,
        )

        print(
            f"  기간 합계: {len(range_rows):,}건"
        )
        print()

    # 전체 기간 통합 파일 저장
    all_filename = (
        f"kat_sale_citrus_"
        f"{start_date.strftime('%Y%m%d')}_"
        f"{end_date.strftime('%Y%m%d')}_all.csv"
    )

    write_csv(
        all_rows,
        OUTPUT_DIR / all_filename,
    )

    # 오류 내역 저장
    write_error_log(
        errors,
        OUTPUT_DIR,
    )

    # 실행 요약 저장
    summary = {
        "api_url": get_api_url(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "large_category_code": LARGE_CATEGORY_CODE,
        "middle_category_code": MIDDLE_CATEGORY_CODE,
        "market_count": len(MARKETS),
        "row_count": len(all_rows),
        "error_count": len(errors),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    summary_path = OUTPUT_DIR / "download_summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("다운로드 작업이 완료되었습니다.")
    print(f"- 전체 데이터: {len(all_rows):,}건")
    print(f"- 오류 건수: {len(errors):,}건")
    print(f"- 출력 폴더: {OUTPUT_DIR.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
