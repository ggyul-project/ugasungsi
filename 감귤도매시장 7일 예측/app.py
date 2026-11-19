from __future__ import annotations

import json
import os
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "rf_dashboard_history.csv"
MANIFEST_PATH = BASE_DIR / "data_manifest.json"
MODEL_PATH = BASE_DIR / "rf_model_bundle.pkl"
FORECASTS_PATH = BASE_DIR / "rf_precomputed_forecasts.csv"
MODEL_INFO_PATH = BASE_DIR / "rf_precomputed_model_info.json"
PRODUCTS = ["조생귤", "한라봉", "하우스감귤", "천혜향"]
WEIGHTS = {
    "조생귤": [5.0, 3.0, 10.0],
    "한라봉": [3.0, 9.0, 5.0],
    "하우스감귤": [3.0, 5.0, 0.5],
    "천혜향": [3.0, 9.0, 5.0],
}
MARKET_LABEL = {
    "서울가락": "서울 가락시장", "대구북부": "대구 북부시장",
    "부산반여": "부산 반여시장", "부산엄궁": "부산 엄궁시장",
    "구리": "경기 구리시장",
}
MARKET_REGION = {
    "서울가락": "서울 송파", "대구북부": "대구 북구",
    "부산반여": "부산 반여", "부산엄궁": "부산 사상",
    "구리": "경기 구리",
}
MARKET_FROM_LABEL = {v: k for k, v in MARKET_LABEL.items()}
MARKET_REGION_BY_LABEL = {MARKET_LABEL[k]: v for k, v in MARKET_REGION.items()}
BASE_KEYS = ["market", "variety", "box_kg"]
VOL_CAT_COLS = ["market", "variety"]
PRICE_CAT_COLS = ["market", "variety", "grade", "size"]
NUM_COLS = [
    "box_kg", "year_offset", "month", "weekday", "doy_sin", "doy_cos",
    "lag_vol_1", "lag_vol_7", "roll_vol_7", "lag_price_1", "lag_price_7", "roll_price_7",
]
VOL_FEATURES = VOL_CAT_COLS + NUM_COLS
PRICE_FEATURES = PRICE_CAT_COLS + NUM_COLS

st.set_page_config(page_title="감귤 도매시장 7일 예측 대시보드", page_icon="🍊", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width:1380px;padding-top:1.4rem;padding-bottom:2.5rem}
    [data-testid="stSidebar"] {border-right:1px solid #e5e5e5}
    div[data-testid="stMetric"] {border:1px solid #e6e6e6;border-radius:8px;padding:12px 14px;min-height:100px}
    [data-testid="stPlotlyChart"] {margin-bottom:.5rem}
    [data-testid="stDataFrame"] {border:1px solid #e6e6e6;border-radius:6px;overflow:hidden}
    [data-testid="stExpander"] {border:1px solid #e6e6e6;border-radius:6px}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f}"


def model_pipeline(kind: str) -> Pipeline:
    cats = VOL_CAT_COLS if kind == "volume" else PRICE_CAT_COLS
    prep = ColumnTransformer(
        [("categories", OneHotEncoder(handle_unknown="ignore"), cats), ("numbers", "passthrough", NUM_COLS)],
        remainder="drop",
    )
    forest = RandomForestRegressor(
        n_estimators=32, min_samples_leaf=4, max_features="sqrt", max_depth=14,
        n_jobs=2, random_state=42,
    )
    return Pipeline([("prep", prep), ("forest", forest)])


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dt = pd.to_datetime(out["date"])
    doy = dt.dt.dayofyear.astype(float)
    out["year_offset"] = dt.dt.year.astype(float) - 2021.0
    out["month"] = dt.dt.month.astype(float)
    out["weekday"] = dt.dt.dayofweek.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("app.py와 rf_dashboard_history.csv, data_manifest.json을 같은 폴더에 두세요.")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["date"])
    for col in ["market", "variety", "grade", "size"]:
        df[col] = df[col].fillna("미지정").astype(str).replace({".": "미지정", "nan": "미지정"})
    for col in ["box_kg", "volume_ton", "price_kg", "volume_kg", "total_price", "price_box", "avgprc_box"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "box_kg", "volume_ton", "price_kg"])
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return df, manifest


def make_training_frames(history: pd.DataFrame):
    last_date = pd.Timestamp(history.date.max()).normalize()
    actual_base = (
        history.groupby(["date"] + BASE_KEYS, as_index=False)
        .agg(volume_ton=("volume_ton", "sum"), volume_kg=("volume_kg", "sum"), total_price=("total_price", "sum"))
        .sort_values(BASE_KEYS + ["date"])
    )
    calendars = []
    for key, group in actual_base.groupby(BASE_KEYS, sort=False):
        index = pd.date_range(group.date.min(), last_date, freq="D")
        g = group.set_index("date").reindex(index)
        g.index.name = "date"
        g["volume_ton"] = g["volume_ton"].fillna(0.0)
        g["volume_kg"] = g["volume_kg"].fillna(0.0)
        observed_price = g["total_price"] / g["volume_kg"].replace(0, np.nan)
        g["base_price"] = observed_price.ffill().fillna(0.0)
        g["lag_vol_1"] = g["volume_ton"].shift(1)
        g["lag_vol_7"] = g["volume_ton"].shift(7)
        g["roll_vol_7"] = g["volume_ton"].shift(1).rolling(7, min_periods=1).mean()
        g["lag_price_1"] = g["base_price"].shift(1)
        g["lag_price_7"] = g["base_price"].shift(7)
        g["roll_price_7"] = g["base_price"].shift(1).rolling(7, min_periods=1).mean()
        g = g.reset_index()
        g["market"], g["variety"], g["box_kg"] = key
        g = add_date_features(g)
        g[NUM_COLS] = g[NUM_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        calendars.append(g)
    base = pd.concat(calendars, ignore_index=True)
    price_feature_cols = [col for col in NUM_COLS if col not in BASE_KEYS]
    price_features = base[["date"] + BASE_KEYS + price_feature_cols]
    price_frame = history.merge(price_features, on=["date"] + BASE_KEYS, how="left", suffixes=("", "_feature"))
    price_frame = add_date_features(price_frame)
    for col in PRICE_CAT_COLS:
        price_frame[col] = price_frame[col].fillna("미지정").astype(str)
    price_frame[NUM_COLS] = price_frame[NUM_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return base, price_frame


def score(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    mask = np.isfinite(actual) & np.isfinite(predicted) & (actual >= 0)
    a, p = actual[mask], predicted[mask]
    positive = a > 0
    mape = float(np.mean(np.abs(a[positive] - p[positive]) / a[positive]) * 100) if positive.any() else np.nan
    return {
        "MAE": float(mean_absolute_error(a, p)) if len(a) else np.nan,
        "RMSE": float(np.sqrt(mean_squared_error(a, p))) if len(a) else np.nan,
        "MAPE": mape,
        "R2": float(r2_score(a, p)) if len(a) > 1 else np.nan,
        "n": int(len(a)),
        "n_positive": int(positive.sum()),
    }


@st.cache_resource(show_spinner=False)
def fit_random_forest(fingerprint: str):
    history, _ = load_data()
    base, price_frame = make_training_frames(history)
    validation_start = base.date.max() - pd.Timedelta(days=45)
    vol_train, vol_valid = base[base.date < validation_start], base[base.date >= validation_start]
    price_train = price_frame[price_frame.date < validation_start]
    price_valid = price_frame[price_frame.date >= validation_start]

    scores = {}
    for kind, train, valid, target, features in [
        ("volume", vol_train, vol_valid, "volume_ton", VOL_FEATURES),
        ("price", price_train, price_valid, "price_kg", PRICE_FEATURES),
    ]:
        model = model_pipeline(kind)
        model.fit(train[features], np.log1p(train[target].clip(lower=0)))
        pred = np.maximum(0, np.expm1(model.predict(valid[features])))
        scores[kind] = score(valid[target], pred)

    models = {}
    for kind, frame, target, features in [
        ("volume", base, "volume_ton", VOL_FEATURES),
        ("price", price_frame, "price_kg", PRICE_FEATURES),
    ]:
        model = model_pipeline(kind)
        model.fit(frame[features], np.log1p(frame[target].clip(lower=0)))
        models[kind] = model

    season = history[["date", "variety"]].drop_duplicates().copy()
    season["year"] = season.date.dt.year
    season["week"] = season.date.dt.isocalendar().week.astype(int)
    yearly = season.groupby(["variety", "week"])["year"].nunique().reset_index(name="years")
    active_weeks = {
        variety: set(yearly[(yearly.variety == variety) & (yearly.years >= 2)].week.astype(int))
        for variety in PRODUCTS
    }
    scores["validation_start"] = validation_start
    scores["validation_end"] = base.date.max()
    scores["volume_train_rows"] = int(len(base))
    scores["price_train_rows"] = int(len(price_frame))
    return models, scores, active_weeks, base


def data_fingerprint(history: pd.DataFrame, manifest: dict) -> str:
    return f"{len(history)}-{history.date.max()}-{manifest['source_member_bytes']}"


@st.cache_resource(show_spinner=False)
def load_model_bundle(fingerprint: str):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"저장된 모델 파일이 없습니다: {MODEL_PATH.name}. train_models.py를 한 번 실행해 주세요.")
    with MODEL_PATH.open("rb") as file:
        bundle = pickle.load(file)
    if bundle.get("fingerprint") != fingerprint:
        raise ValueError("저장된 모델이 현재 CSV와 맞지 않습니다. train_models.py로 모델 파일을 다시 만들어 주세요.")
    return bundle["models"], bundle["scores"], bundle["active_weeks"], bundle["base"]


@st.cache_data(show_spinner=False)
def load_precomputed_forecasts():
    if not FORECASTS_PATH.exists():
        raise FileNotFoundError(f"미리 계산한 예측 데이터가 없습니다: {FORECASTS_PATH.name}")
    frame = pd.read_csv(FORECASTS_PATH, parse_dates=["date", "forecast_start"])
    frame["market"] = frame.market.astype(str)
    frame["variety"] = frame.variety.astype(str)
    frame["box_kg"] = pd.to_numeric(frame.box_kg, errors="coerce")
    return frame


@st.cache_data(show_spinner=False)
def load_precomputed_model_info():
    if not MODEL_INFO_PATH.exists():
        raise FileNotFoundError(f"모델 정보 파일이 없습니다: {MODEL_INFO_PATH.name}")
    return json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))


if os.environ.get("CITRUS_TRAIN_MODEL_ONLY") == "1":
    train_history, train_manifest = load_data()
    train_fingerprint = data_fingerprint(train_history, train_manifest)
    trained_models, train_scores, train_active_weeks, train_base = fit_random_forest(train_fingerprint)
    model_bundle = {
        "fingerprint": train_fingerprint,
        "models": trained_models,
        "scores": train_scores,
        "active_weeks": train_active_weeks,
        "base": train_base,
    }
    with MODEL_PATH.open("wb") as file:
        pickle.dump(model_bundle, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved trained RandomForest bundle: {MODEL_PATH} ({MODEL_PATH.stat().st_size:,} bytes)")
    raise SystemExit(0)


def forest_values(model: Pipeline, frame: pd.DataFrame, features: list[str]):
    transformed = model.named_steps["prep"].transform(frame[features])
    forest = model.named_steps["forest"]
    trees = np.vstack([tree.predict(transformed) for tree in forest.estimators_])
    actual_scale = np.maximum(0, np.expm1(trees))
    mean = np.maximum(0, np.expm1(model.predict(frame[features])))
    return mean, np.quantile(actual_scale, 0.10, axis=0), np.quantile(actual_scale, 0.90, axis=0)


HOLIDAYS = {
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-09-24", "2026-09-25", "2026-09-26",
    "2026-12-25", "2027-01-01", "2027-02-06", "2027-02-07", "2027-02-08",
}
for year in [2026, 2027]:
    for md in ["01-01", "03-01", "05-05", "06-06", "08-15", "10-03", "10-09", "12-25"]:
        HOLIDAYS.add(f"{year}-{md}")


def is_model_open(day: date, active_weeks: set[int]) -> bool:
    week = int(pd.Timestamp(day).isocalendar().week)
    return day.weekday() != 6 and day.isoformat() not in HOLIDAYS and week in active_weeks


def open_days(start: date, active_weeks: set[int], count: int = 7):
    days, cursor, limit = [], start, start + timedelta(days=75)
    while cursor <= limit and len(days) < count:
        if is_model_open(cursor, active_weeks):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def create_states(base: pd.DataFrame):
    states = {}
    for key, group in base.groupby(BASE_KEYS, sort=False):
        group = group.sort_values("date")
        states[key] = {
            "dates": list(group.date),
            "volumes": group.volume_ton.astype(float).tolist(),
            "prices": group.base_price.astype(float).tolist(),
        }
    return states


def features_for(day: date, market: str, variety: str, box_kg: float, state: dict) -> dict:
    volumes, prices = state["volumes"], state["prices"]
    dt = pd.Timestamp(day)
    doy = float(dt.dayofyear)
    out = {
        "date": dt,
        "market": market,
        "variety": variety,
        "box_kg": float(box_kg),
        "year_offset": float(dt.year - 2021),
        "month": float(dt.month),
        "weekday": float(dt.dayofweek),
        "doy_sin": float(np.sin(2 * np.pi * doy / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * doy / 365.25)),
        "lag_vol_1": volumes[-1] if volumes else 0.0,
        "lag_vol_7": volumes[-7] if len(volumes) >= 7 else float(np.mean(volumes)) if volumes else 0.0,
        "roll_vol_7": float(np.mean(volumes[-7:])) if volumes else 0.0,
        "lag_price_1": prices[-1] if prices else 0.0,
        "lag_price_7": prices[-7] if len(prices) >= 7 else float(np.mean(prices)) if prices else 0.0,
        "roll_price_7": float(np.mean(prices[-7:])) if prices else 0.0,
    }
    return out


def seasonal_shares(source: pd.DataFrame, day: date) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame(columns=["grade", "size", "share"])
    target_doy = pd.Timestamp(day).dayofyear
    doy = source.date.dt.dayofyear.to_numpy(dtype=float)
    delta = np.abs((doy - target_doy + 182.625) % 365.25 - 182.625)
    seasonal = pd.DataFrame()
    for radius in [21.0, 42.0, 84.0, 365.25]:
        seasonal = source.loc[delta <= radius]
        if not seasonal.empty and seasonal.volume_kg.sum() > 0:
            break
    weights = (
        seasonal.groupby(["grade", "size"], as_index=False)
        .agg(season_kg=("volume_kg", "sum"))
    )
    if weights.empty or weights.season_kg.sum() <= 0:
        return pd.DataFrame(columns=["grade", "size", "share"])
    weights["share"] = weights.season_kg / weights.season_kg.sum()
    return weights[["grade", "size", "share"]]


def simulate_market(history, models, state, market, variety, box_kg, sim_start, target_dates, active_weeks, grade_filter, size_filter):
    source = history[
        (history.market == market) & (history.variety == variety) & (history.box_kg == float(box_kg))
    ].copy()
    if source.empty:
        return []
    final_day = target_dates[-1]
    target_set = {pd.Timestamp(d) for d in target_dates}
    rows = []
    cursor = sim_start
    while cursor <= final_day:
        base_row = features_for(cursor, market, variety, box_kg, state)
        shares = seasonal_shares(source, cursor)
        if not is_model_open(cursor, active_weeks) or shares.empty:
            total_volume = lo_volume = hi_volume = 0.0
            average_price = state["prices"][-1] if state["prices"] else 0.0
            selected_share = 0.0 if shares.empty else 1.0
            selected_price = average_price
            price_lo = price_hi = average_price
        else:
            vol_frame = pd.DataFrame([base_row])
            total_v, v10, v90 = forest_values(models["volume"], vol_frame, VOL_FEATURES)
            total_volume, lo_volume, hi_volume = float(total_v[0]), float(v10[0]), float(v90[0])

            price_frame = shares.copy()
            for col, value in base_row.items():
                price_frame[col] = value
            price_frame["grade"] = price_frame.grade.fillna("미지정").astype(str)
            price_frame["size"] = price_frame["size"].fillna("미지정").astype(str)
            pred, p10, p90 = forest_values(models["price"], price_frame, PRICE_FEATURES)
            price_frame["pred_price"] = pred
            price_frame["p10"] = p10
            price_frame["p90"] = p90
            weights = price_frame.share.to_numpy(dtype=float)
            average_price = float(np.average(price_frame.pred_price, weights=weights))

            visible = price_frame.copy()
            if grade_filter != "전체 등급":
                visible = visible[visible.grade == grade_filter]
            if size_filter != "전체 규격":
                visible = visible[visible["size"] == size_filter]
            selected_share = float(visible.share.sum())
            if visible.empty or selected_share <= 0:
                selected_share, selected_price = 0.0, average_price
                price_lo = price_hi = average_price
            else:
                w = visible.share.to_numpy(dtype=float)
                selected_price = float(np.average(visible.pred_price, weights=w))
                price_lo = float(np.average(visible.p10, weights=w))
                price_hi = float(np.average(visible.p90, weights=w))

        # Append a forecast for every calendar date so a future start date has realistic lag inputs.
        state["dates"].append(pd.Timestamp(cursor))
        state["volumes"].append(float(total_volume))
        state["prices"].append(float(average_price))
        if pd.Timestamp(cursor) in target_set:
            rows.append({
                "date": pd.Timestamp(cursor), "market": market,
                "volume_ton": float(total_volume * selected_share),
                "volume_total_ton": float(total_volume),
                "volume_p10": float(lo_volume * selected_share),
                "volume_p90": float(hi_volume * selected_share),
                "price_kg": float(selected_price),
                "box_price": float(selected_price * box_kg),
                "price_p10": float(price_lo), "price_p90": float(price_hi),
                "grade_share": selected_share,
            })
        cursor += timedelta(days=1)
    return rows


def actuals_for(history, markets, variety, box_kg, grade, size, start, end):
    end_ts = pd.Timestamp(end)
    sub = history[
        history.market.isin(markets) & (history.variety == variety) & (history.box_kg == box_kg)
        & (history.date >= pd.Timestamp(start)) & (history.date <= end_ts)
    ].copy()
    if grade != "전체 등급":
        sub = sub[sub.grade == grade]
    if size != "전체 규격":
        sub = sub[sub["size"] == size]
    if sub.empty:
        return sub
    daily = sub.groupby(["date", "market"], as_index=False).agg(
        volume_ton=("volume_ton", "sum"), volume_kg=("volume_kg", "sum"), total_price=("total_price", "sum")
    )
    daily["price_kg"] = daily.total_price / daily.volume_kg.replace(0, np.nan)
    return daily


def market_summary(frame, target_date, box_kg, value_label="예측"):
    rows = []
    for market, sub in frame.groupby("market", sort=True):
        sub = sub.sort_values("date")
        at_start = sub[sub.date.dt.date == target_date]
        first = float(at_start.price_kg.iloc[0]) if len(at_start) else float(sub.price_kg.iloc[0])
        avg = float(np.average(sub.price_kg, weights=sub.volume_ton.clip(lower=.0001)))
        total = float(sub.volume_ton.sum())
        change = (float(sub.price_kg.iloc[-1]) / float(sub.price_kg.iloc[0]) - 1) * 100 if len(sub) > 1 else 0.0
        rows.append({
            "도매시장": market,
            "권역": MARKET_REGION.get(
                market,
                "전체 시장" if market == "전체 시장" else
                "주요 5개 시장" if market == "주요 5개 시장" else
                "전체 시장" if market == "주요 5개 시장 전체" else
                MARKET_REGION_BY_LABEL.get(market, "-"),
            ),
            f"선택일 {value_label} 가격 (원/kg)": round(first),
            "선택일 상자 가격 (원)": round(first * box_kg),
            f"7일 평균 {value_label} 가격 (원/kg)": round(avg),
            "7일 가격 변동 (%)": round(change, 1),
            f"7일 총 {value_label} 거래량 (톤)": round(total, 2),
            "일평균 거래량 (톤)": round(total / max(1, len(sub)), 2),
        })
    return pd.DataFrame(rows).sort_values("도매시장").reset_index(drop=True)


def aggregate_market_frame(frame, market_ids, label):
    subset = frame[frame.market.isin(market_ids)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["date", "market", "volume_ton", "price_kg"])
    weight_col = "volume_kg" if "volume_kg" in subset.columns else "volume_ton"
    subset["weighted_price"] = subset.price_kg * subset[weight_col]
    daily = subset.groupby("date", as_index=False).agg(
        volume_ton=("volume_ton", "sum"),
        weighted_price=("weighted_price", "sum"),
        price_weight=(weight_col, "sum"),
    )
    daily["price_kg"] = daily.weighted_price / daily.price_weight.replace(0, np.nan)
    daily["price_kg"] = daily.price_kg.fillna(0.0)
    daily["market"] = label
    return daily[["date", "market", "volume_ton", "price_kg"]]


def export_precomputed_forecasts():
    history, manifest = load_data()
    fingerprint = data_fingerprint(history, manifest)
    models, scores, active_weeks, base = load_model_bundle(fingerprint)
    markets = sorted(history.market.dropna().astype(str).unique())
    last_actual = history.date.max().date()
    generated_on = date.today()

    # Precompute the manual date range and the two fast date choices.
    start_dates = sorted(
        {generated_on + timedelta(days=offset) for offset in range(-3, 4)}
        | {generated_on + timedelta(days=7)}
    )
    states = create_states(base)
    exported = []
    for variety in PRODUCTS:
        active = active_weeks.get(variety, set())
        days_by_start = {
            start: open_days(start, active, 7)
            for start in start_dates
            if start > last_actual
        }
        target_days = sorted({day for days in days_by_start.values() for day in days})
        if not target_days:
            continue
        for box_kg in WEIGHTS[variety]:
            for market in markets:
                state = states.get((market, variety, float(box_kg)))
                if state is None:
                    continue
                rows = simulate_market(
                    history, models, state, market, variety, float(box_kg),
                    last_actual + timedelta(days=1), target_days, active,
                    "전체 등급", "전체 규격",
                )
                if not rows:
                    continue
                daily = pd.DataFrame(rows)
                for start, forecast_days in days_by_start.items():
                    selected = daily[daily.date.dt.date.isin(forecast_days)].copy()
                    if selected.empty:
                        continue
                    selected["forecast_start"] = pd.Timestamp(start)
                    selected["variety"] = variety
                    selected["box_kg"] = float(box_kg)
                    exported.append(selected)

    if not exported:
        raise RuntimeError("저장할 예측 결과가 없습니다. 모델 출하기와 날짜 범위를 확인해 주세요.")
    output = pd.concat(exported, ignore_index=True)
    output.to_csv(FORECASTS_PATH, index=False, encoding="utf-8-sig")

    model_info = {
        "generated_on": generated_on.isoformat(),
        "history_last_date": last_actual.isoformat(),
        "market_count": int(history.market.nunique()),
        "forecast_start_min": start_dates[0].isoformat(),
        "forecast_start_max": start_dates[-1].isoformat(),
        "forecast_rows": int(len(output)),
        "validation_start": pd.Timestamp(scores["validation_start"]).date().isoformat(),
        "validation_end": pd.Timestamp(scores["validation_end"]).date().isoformat(),
        "volume_train_rows": int(scores["volume_train_rows"]),
        "price_train_rows": int(scores["price_train_rows"]),
        "scores": {
            key: {metric: float(value) if metric != "n" and metric != "n_positive" else int(value)
                  for metric, value in scores[key].items()}
            for key in ["price", "volume"]
        },
    }
    MODEL_INFO_PATH.write_text(json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(output):,} forecast rows to {FORECASTS_PATH}")
    print(f"Saved validation details to {MODEL_INFO_PATH}")


if os.environ.get("CITRUS_PRECOMPUTE_ONLY") == "1":
    export_precomputed_forecasts()
    raise SystemExit(0)


try:
    history, manifest = load_data()
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다: {exc}")
    st.stop()

MARKETS = manifest["top5_markets"]
ALL_MARKETS = sorted(history.market.dropna().astype(str).unique())
# Keep readable labels for the original five, and expose every other market
# using the exact name recorded in the supplied data.
KNOWN_MARKET_LABELS = MARKET_LABEL.copy()
KNOWN_MARKET_REGIONS = MARKET_REGION.copy()
MARKET_LABEL = {market: KNOWN_MARKET_LABELS.get(market, market) for market in ALL_MARKETS}
MARKET_REGION = {market: KNOWN_MARKET_REGIONS.get(market, "-") for market in ALL_MARKETS}
MARKET_FROM_LABEL = {v: k for k, v in MARKET_LABEL.items()}
MARKET_REGION_BY_LABEL = {MARKET_LABEL[k]: v for k, v in MARKET_REGION.items()}
today = date.today()
last_actual_date = history.date.max().date()
picker_min = today - timedelta(days=3)
picker_max = today + timedelta(days=3)
quick_week_date = today + timedelta(days=7)
default_date = today


def clear_quick_date_override():
    st.session_state.pop("quick_target_date", None)


saved_picker_date = st.session_state.get("forecast_date")
if saved_picker_date is not None:
    saved_picker_date = pd.Timestamp(saved_picker_date).date()
    if not picker_min <= saved_picker_date <= picker_max:
        st.session_state["forecast_date"] = default_date

with st.sidebar:
    st.markdown("### 🔎 조회 조건 설정")
    market_options = ["주요 5개 시장 함께 보기"] + [MARKET_LABEL[m] for m in ALL_MARKETS]
    market_label = st.selectbox(
        f"시장 선택 ({len(ALL_MARKETS)}개 시장)", market_options,
        index=1 + ALL_MARKETS.index("서울가락") if "서울가락" in ALL_MARKETS else 1,
        help="개별 시장은 데이터에 있는 모든 시장 중에서 고를 수 있습니다. 함께 보기를 고르면 전체 시장과 주요 5개 시장을 비교합니다.",
        key="market_choice",
    )
    variety = st.selectbox("품목 선택", PRODUCTS, key="variety_choice")
    box_kg = st.selectbox("거래단위 선택", WEIGHTS[variety], format_func=lambda x: f"{x:g}kg 상자", key="box_choice")
    chosen_market = None if market_label == "주요 5개 시장 함께 보기" else MARKET_FROM_LABEL[market_label]
    scope_markets = MARKETS if chosen_market is None else [chosen_market]
    available = history[
        history.market.isin(scope_markets) & (history.variety == variety) & (history.box_kg == float(box_kg))
    ]
    # Grade and size are intentionally not exposed as filters. Predictions
    # aggregate all historical grade/size combinations for the selected item.
    grade = "전체 등급"
    size = "전체 규격"
    if "forecast_date" not in st.session_state:
        st.session_state["forecast_date"] = default_date
    st.caption(f"빠른 날짜 · 오늘 {today:%m/%d}")
    b1, b2 = st.columns(2)
    quick_buttons = [(b1, f"오늘 ({today:%m/%d})", today), (b2, "+7일", quick_week_date)]
    for i, (col, label, chosen) in enumerate(quick_buttons):
        if col.button(label, key=f"quick_{i}", width="stretch"):
            st.session_state["quick_target_date"] = chosen
    picker_date = st.date_input(
        "날짜 선택 (오늘 기준 3일 전~3일 후)", min_value=picker_min, max_value=picker_max,
        key="forecast_date", on_change=clear_quick_date_override,
    )
    target_date = st.session_state.get("quick_target_date", picker_date)
    st.caption(f"조회 기준일: {target_date:%Y-%m-%d}")
    if st.button("↻ 필터 초기화", width="stretch"):
        for key in ["market_choice", "variety_choice", "box_choice", "forecast_date", "quick_target_date"]:
            st.session_state.pop(key, None)
        st.rerun()

st.title("🍊 감귤 도매시장 7일 예측")
st.caption(f"RandomForest · 제공된 병합 CSV 한 파일 · 자료 기간 {history.date.min():%Y-%m-%d} ~ {last_actual_date:%Y-%m-%d}")
st.divider()

with st.spinner("미리 계산해 저장한 예측 데이터를 불러오는 중입니다."):
    try:
        precomputed_forecasts = load_precomputed_forecasts()
        model_info = load_precomputed_model_info()
        scores = model_info["scores"]
    except Exception as exc:
        st.error(f"저장된 예측 데이터를 불러오지 못했습니다: {exc}")
        st.stop()

is_historical_selection = target_date <= last_actual_date
if is_historical_selection:
    historical = actuals_for(
        history, ALL_MARKETS, variety, float(box_kg), grade, size, target_date, last_actual_date,
    )
    dates = sorted(historical.date.dt.date.unique())[:7] if not historical.empty else []
    if not dates:
        st.warning("선택한 날짜 이후에 이 품목의 실거래 자료가 없습니다.")
        st.stop()
    forecast = historical[historical.date.dt.date.isin(dates)].copy()
    forecast["volume_total_ton"] = forecast.volume_ton
    forecast["volume_p10"] = forecast.volume_ton
    forecast["volume_p90"] = forecast.volume_ton
    forecast["box_price"] = forecast.price_kg * float(box_kg)
    forecast["price_p10"] = forecast.price_kg
    forecast["price_p90"] = forecast.price_kg
    forecast["grade_share"] = 1.0
    value_label = "실거래"
else:
    forecast = precomputed_forecasts[
        (precomputed_forecasts.forecast_start.dt.date == target_date)
        & (precomputed_forecasts.market.isin(ALL_MARKETS))
        & (precomputed_forecasts.variety == variety)
        & (precomputed_forecasts.box_kg == float(box_kg))
    ].copy()
    if forecast.empty:
        st.warning(
            f"{target_date:%Y-%m-%d} 예측 데이터가 저장 파일에 없습니다. "
            "같은 폴더에서 precompute_forecasts.py를 실행해 예측 데이터를 갱신해 주세요."
        )
        st.stop()
    dates = sorted(forecast.date.dt.date.unique())
    value_label = "예측"

forecast["weekday"] = forecast.date.dt.dayofweek.map({0:"월",1:"화",2:"수",3:"목",4:"금",5:"토",6:"일"})
selected_rows = forecast[forecast.market.isin(MARKETS)] if chosen_market is None else forecast[forecast.market == chosen_market]
if selected_rows.empty:
    selected_rows = forecast
target_rows = selected_rows[selected_rows.date.dt.date == target_date]
if target_rows.empty:
    target_rows = selected_rows[selected_rows.date == selected_rows.date.min()]
first_price = float(np.average(target_rows.price_kg, weights=target_rows.volume_ton.clip(lower=.0001)))
daily_rollup = selected_rows.groupby("date").apply(
    lambda g: np.average(g.price_kg, weights=g.volume_ton.clip(lower=.0001)), include_groups=False
)
mean_price = float(np.average(selected_rows.price_kg, weights=selected_rows.volume_ton.clip(lower=.0001)))
price_change = (float(daily_rollup.iloc[-1]) / float(daily_rollup.iloc[0]) - 1) * 100 if len(daily_rollup) > 1 else 0.0
total_volume = float(selected_rows.volume_ton.sum())
main_market = chosen_market or "서울가락"
main_market_name = MARKET_LABEL[main_market]
overview_market_name = main_market_name if chosen_market else "주요 5개 시장"
latest_sub = available.copy()
if chosen_market is not None:
    latest_sub = latest_sub[latest_sub.market == chosen_market]
latest_item_date = latest_sub.date.max().date() if len(latest_sub) else None

st.markdown(f"**{overview_market_name} · {variety} {box_kg:g}kg**")
st.caption(f"전체 등급·크기 기준 · {value_label} 기간 {dates[0]} ~ {dates[-1]} ({len(dates)}개 거래일)")
k1, k2, k3 = st.columns(3, gap="medium")
k1.metric("선택일 상자 가격", f"{fmt(first_price * float(box_kg))}원")
k1.caption(f"단가 {fmt(first_price)}원/kg")
k2.metric("7일 평균 상자 가격", f"{fmt(mean_price * float(box_kg))}원")
k2.caption(f"선택일 대비 {price_change:+.1f}%")
k3.metric(f"7일 {value_label} 거래량", f"{fmt(total_volume, 1)}톤")
k3.caption(f"일평균 {fmt(total_volume / max(1, len(dates) * selected_rows.market.nunique()), 1)}톤")
latest_label = latest_item_date.strftime("%Y-%m-%d") if latest_item_date else "자료 없음"
st.caption(f"최근 실거래 기준일: {latest_label}")

main_sub = forecast[forecast.market == main_market].sort_values("date")
if main_sub.empty:
    main_sub = forecast.sort_values("date")
    main_market_name = MARKET_LABEL[main_sub.market.iloc[0]]
st.subheader(f"7일 가격·거래량 {value_label}")
st.caption("가격과 거래량은 단위가 달라 각각 나눠 표시합니다.")
price_low, price_high = float(main_sub.price_kg.min()), float(main_sub.price_kg.max())
price_pad = max((price_high - price_low) * .2, price_high * .025, 20.0)
price_fig = go.Figure(go.Scatter(
    x=main_sub.date, y=main_sub.price_kg, name=f"{value_label} 가격", mode="lines+markers",
    line=dict(color="#e58a42", width=3), marker=dict(size=8),
    customdata=np.c_[main_sub.price_p10, main_sub.price_p90],
    hovertemplate=(
        "%{x|%Y-%m-%d}<br>실거래 가격 %{y:,.0f}원/kg<extra></extra>"
        if is_historical_selection else
        "%{x|%Y-%m-%d}<br>예측 가격 %{y:,.0f}원/kg<br>참고 범위 %{customdata[0]:,.0f}~%{customdata[1]:,.0f}원/kg<extra></extra>"
    ),
))
price_fig.update_layout(
    height=310, margin=dict(l=10,r=16,t=20,b=15), showlegend=False,
    plot_bgcolor="#ffffff", paper_bgcolor="rgba(255,255,255,0)", hovermode="x unified",
)
price_fig.update_xaxes(title=None, tickformat="%m/%d", showgrid=False)
price_fig.update_yaxes(title="원/kg", range=[max(0, price_low - price_pad), price_high + price_pad], gridcolor="#e9e9e9")

volume_fig = go.Figure(go.Bar(
    x=main_sub.date, y=main_sub.volume_ton, name=f"{value_label} 거래량", marker_color="#65a995",
    hovertemplate="%{x|%Y-%m-%d}<br>거래량 %{y:,.2f}톤<extra></extra>",
))
volume_fig.update_layout(
    height=310, margin=dict(l=10,r=16,t=20,b=15), showlegend=False,
    plot_bgcolor="#ffffff", paper_bgcolor="rgba(255,255,255,0)", bargap=.35,
)
volume_fig.update_xaxes(title=None, tickformat="%m/%d", showgrid=False)
volume_fig.update_yaxes(title="톤", rangemode="tozero", gridcolor="#e9e9e9")
price_col, volume_col = st.columns(2, gap="large")
with price_col:
    st.markdown(f"**{main_market_name} · 가격 (원/kg)**")
    st.plotly_chart(price_fig, use_container_width=True, config={"displayModeBar": False})
with volume_col:
    st.markdown(f"**{main_market_name} · 거래량 (톤)**")
    st.plotly_chart(volume_fig, use_container_width=True, config={"displayModeBar": False})

market_colors = {}
if chosen_market is None:
    st.subheader("주요 5개 시장 비교")
    st.caption("전체 시장 집계는 제외하고, 주요 5개 시장을 각각 표시합니다.")
    comparison = forecast[forecast.market.isin(MARKETS)].copy()
    market_palette = ["#267451", "#e28a3b", "#4e79a7", "#9b6c9e", "#c74f53"]
    market_colors = {MARKET_LABEL[m]: color for m, color in zip(MARKETS, market_palette)}
else:
    st.subheader("선택 시장과 전체 시장 비교")
    st.caption(f"{MARKET_LABEL[chosen_market]}와 데이터에 있는 전체 {len(ALL_MARKETS)}개 시장 평균을 비교합니다.")
    selected_label = MARKET_LABEL[chosen_market]
    market_colors = {selected_label: "#267451", "전체 시장": "#929a95"}
compare_mode = st.radio("비교 그래프", ["가격 추이", "거래량 비교"], horizontal=True, label_visibility="collapsed", key="compare_mode")
if chosen_market is not None:
    selected_comparison = forecast[forecast.market == chosen_market].copy()
    selected_comparison["market"] = selected_label
    overall_comparison = aggregate_market_frame(forecast, ALL_MARKETS, "전체 시장")
    comparison = pd.concat([selected_comparison, overall_comparison], ignore_index=True)
comparison["시장"] = comparison.market.map(lambda market: MARKET_LABEL.get(market, market))
if compare_mode == "가격 추이":
    compare_fig = px.line(comparison, x="date", y="price_kg", color="시장", markers=True,
    labels={"date":f"{value_label}일", "price_kg":f"{value_label} 단가 (원/kg)", "시장":"도매시장"}, color_discrete_map=market_colors)
    compare_fig.update_traces(line=dict(width=2.7), marker=dict(size=7))
    compare_low, compare_high = float(comparison.price_kg.min()), float(comparison.price_kg.max())
    compare_pad = max((compare_high - compare_low) * .15, compare_high * .02, 20.0)
    compare_fig.update_yaxes(title=f"{value_label} 단가 (원/kg)", range=[max(0, compare_low - compare_pad), compare_high + compare_pad], gridcolor="#edf1ed")
else:
    compare_fig = px.bar(comparison, x="date", y="volume_ton", color="시장", barmode="group",
        labels={"date":f"{value_label}일", "volume_ton":f"{value_label} 거래량 (톤)", "시장":"도매시장"}, color_discrete_map=market_colors)
    compare_fig.update_yaxes(title=f"{value_label} 거래량 (톤)", rangemode="tozero", gridcolor="#edf1ed")
compare_fig.update_layout(
    height=350, margin=dict(l=12,r=14,t=28,b=18), plot_bgcolor="#ffffff", paper_bgcolor="rgba(255,255,255,0)",
    font=dict(family="Arial, sans-serif", color="#52665b"), legend=dict(orientation="h", y=1.12, x=.02),
)
compare_fig.update_xaxes(title=None, tickformat="%m/%d", showgrid=False, linecolor="#dfe8e1")
st.plotly_chart(compare_fig, use_container_width=True, config={"displayModeBar": False})

summary = market_summary(comparison, target_date, float(box_kg), value_label)
st.subheader(f"주요 5개 시장 {value_label} 요약" if chosen_market is None else
             f"선택 시장·전체 시장 {value_label} 요약")
st.caption("선택일 가격과 7일 누적 거래량을 비교합니다.")
st.dataframe(summary, width="stretch", hide_index=True, column_config={
    f"선택일 {value_label} 가격 (원/kg)": st.column_config.NumberColumn(format="%d원/kg"),
    "선택일 상자 가격 (원)": st.column_config.NumberColumn(format="%d원"),
    f"7일 평균 {value_label} 가격 (원/kg)": st.column_config.NumberColumn(format="%d원/kg"),
    "7일 가격 변동 (%)": st.column_config.NumberColumn(format="%.1f%%"),
    f"7일 총 {value_label} 거래량 (톤)": st.column_config.NumberColumn(format="%.2f톤"),
    "일평균 거래량 (톤)": st.column_config.NumberColumn(format="%.2f톤"),
})
st.download_button(f"{value_label} 요약 CSV 저장", summary.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"citrus_market_{'actuals' if is_historical_selection else 'forecast'}_summary.csv", mime="text/csv")

st.subheader(f"{main_market_name} 일별 {value_label}")
st.caption(f"{variety} · {box_kg:g}kg · 전체 등급·크기 기준")
detail = main_sub.copy().sort_values("date")
detail["delta"] = detail.price_kg.diff()
detail_table = pd.DataFrame({
    "날짜": detail.date.dt.strftime("%Y-%m-%d"), "요일": detail.weekday,
    f"{value_label} 가격 (원/kg)": detail.price_kg.round().astype(int),
    f"{value_label} 상자 가격 (원)": detail.box_price.round().astype(int),
    "전일 대비 (원/kg)": detail.delta.round(),
    f"{value_label} 거래량 (톤)": detail.volume_ton.round(2),
})
if not is_historical_selection:
    detail_table["참고 가격 범위 (원/kg)"] = [
        f"{a:,.0f} ~ {b:,.0f}" for a, b in zip(detail.price_p10, detail.price_p90)
    ]
    st.caption("가격 범위는 RandomForest 개별 트리 예측값의 10~90 백분위 참고 범위이며, 통계적 보증구간은 아닙니다.")
st.dataframe(detail_table, width="stretch", hide_index=True, column_config={
    f"{value_label} 가격 (원/kg)": st.column_config.NumberColumn(format="%d원/kg"),
    f"{value_label} 상자 가격 (원)": st.column_config.NumberColumn(format="%d원"),
    "전일 대비 (원/kg)": st.column_config.NumberColumn(format="%+d원/kg"),
    f"{value_label} 거래량 (톤)": st.column_config.NumberColumn(format="%.2f톤"),
})
st.download_button(f"선택 시장 일별 {value_label} CSV 저장", detail_table.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"citrus_7day_{'actuals' if is_historical_selection else 'forecast'}.csv", mime="text/csv")

actuals_end = history.date.max().date()
actuals_start = actuals_end - timedelta(days=6)
actuals = actuals_for(history, ALL_MARKETS, variety, float(box_kg), grade, size, actuals_start, actuals_end)
if not actuals.empty:
    if chosen_market is None:
        actuals = actuals[actuals.market.isin(MARKETS)].copy()
        actuals["시장"] = actuals.market.map(MARKET_LABEL)
    else:
        actuals_selected = actuals[actuals.market == chosen_market].copy()
        actuals_selected["market"] = MARKET_LABEL[chosen_market]
        actuals_overall = aggregate_market_frame(actuals, ALL_MARKETS, "전체 시장")
        actuals = pd.concat([actuals_selected, actuals_overall], ignore_index=True)
        actuals["시장"] = actuals.market
    st.subheader("최근 7일 실제 거래")
    st.caption(
        "데이터의 마지막 실거래일 기준 최근 7일입니다. 주요 5개 시장을 각각 표시합니다."
        if chosen_market is None else
        f"데이터의 마지막 실거래일 기준 최근 7일입니다. {MARKET_LABEL[chosen_market]}와 전체 시장을 비교합니다."
    )
    actual_fig = px.line(actuals, x="date", y="price_kg", color="시장", markers=True,
        labels={"date":"거래일", "price_kg":"실제 단가 (원/kg)", "시장":"도매시장"}, color_discrete_map=market_colors)
    actual_fig.update_layout(height=330, margin=dict(l=12,r=14,t=28,b=18), plot_bgcolor="#ffffff", paper_bgcolor="rgba(255,255,255,0)", legend=dict(orientation="h", y=1.12), font=dict(family="Arial, sans-serif", color="#52665b"))
    actual_fig.update_xaxes(title=None, tickformat="%m/%d", showgrid=False, linecolor="#dfe8e1")
    actual_low, actual_high = float(actuals.price_kg.min()), float(actuals.price_kg.max())
    actual_pad = max((actual_high - actual_low) * .1, actual_high * .02, 20.0)
    actual_fig.update_yaxes(title="실제 단가 (원/kg)", range=[max(0, actual_low - actual_pad), actual_high + actual_pad], gridcolor="#edf1ed")
    st.plotly_chart(actual_fig, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("마지막 실거래일 기준 최근 7일에 선택 조건으로 집계할 실제 거래가 없습니다.")

with st.expander("모델 검증 결과 및 데이터 사용 범위"):
    a, b, c, d = st.columns(4)
    a.metric("적용 모델", "RandomForest")
    b.metric("가격 검증 MAPE", f"{scores['price']['MAPE']:.1f}%")
    c.metric("거래량 검증 MAPE", f"{scores['volume']['MAPE']:.1f}%")
    d.metric("가격 검증 표본", f"{scores['price']['n']:,}행")
    if scores["volume"]["MAPE"] >= 100:
        st.warning("시간순 검증에서 거래량 MAPE가 100%를 넘었습니다. 거래량 전망은 참고값으로 보고 실제 의사결정 전에 추가 검증이 필요합니다.")
    elif scores["volume"]["MAPE"] >= 50:
        st.warning("시간순 검증에서 거래량 오차가 큰 편입니다. 거래량 전망은 참고값으로 확인해 주세요.")
    st.write(f"미리 계산한 예측 파일 생성일: {model_info['generated_on']} · 검증 기간: {pd.Timestamp(model_info['validation_start']):%Y-%m-%d} ~ {pd.Timestamp(model_info['validation_end']):%Y-%m-%d}.")
    st.write(f"가격 MAE {scores['price']['MAE']:,.0f}원/kg · 거래량 MAE {scores['volume']['MAE']:.2f}톤/일 · 거래량 MAPE는 거래가 있었던 날 {scores['volume']['n_positive']:,}개에서 계산.")
    st.write(f"학습 행: 일별 전체거래량 {model_info['volume_train_rows']:,}행(거래 없는 날은 0톤 포함), 세부 단가 {model_info['price_train_rows']:,}행.")
    st.write("거래량은 시장·품목·상자 단위로 먼저 예측한 뒤, 과거 같은 계절의 등급·크기 점유율을 적용했습니다. 가격은 세부 등급·크기 거래를 학습합니다.")
    st.write("Streamlit 화면은 저장된 예측 CSV만 읽습니다. 모델 학습과 예측 계산은 화면 실행 중 수행하지 않습니다.")
    st.write("거래량은 unit_tot_qty 합계를 톤으로 환산하고, 단가는 totprc 합계 ÷ unit_tot_qty 합계로 원/kg를 계산했습니다.")
    st.write(f"전체 학습 시장: {len(ALL_MARKETS)}개 · 주요 5개 시장: {', '.join(MARKET_LABEL[m] for m in MARKETS)}. 가격 위치 순위는 표시하지 않습니다.")

with st.expander("주차 진행 내용 및 입력되지 않은 항목"):
    st.markdown("""
    - 적용 모델: RandomForest
    - 상자 규격: 조생귤 3·5·10kg, 한라봉 3·5·9kg, 하우스감귤 0.5·3·5kg, 천혜향 3·5·9kg
    - XGBoost·TFT·LightGBM 비교값과 1~5차 개선율은 제공된 문서에 수치가 없어 임의로 채우지 않았습니다.
    """)

with st.expander("원본 파일 조사 내역"):
    st.write(f"사용 원본: {manifest['source_archive']} 안의 {manifest['source_member_used']} 한 파일.")
    st.write(f"원본 {manifest['source_rows']:,}행, {manifest['source_market_count']}개 시장, {manifest['source_date_min']}~{manifest['source_date_max']}.")
    st.write(f"지정 품목·상자 중량 {manifest['selected_raw_rows']:,}행; 정확 중복 {manifest['selected_exact_duplicate_rows']:,}행.")
    valid_unique_rows = manifest.get("valid_unique_rows")
    valid_rows_text = f"{int(valid_unique_rows):,}" if valid_unique_rows is not None else "자료에 미기록"
    daily_rows = manifest.get("daily_rows", manifest.get("valid_daily_rows", len(history)))
    st.write(f"전체 시장 유효 원본 {valid_rows_text}행을 {int(daily_rows):,}개 일별 집계로 변환했습니다.")
    st.write("ZIP 안의 part1/part2 파일과 병합 CSV는 같은 데이터의 분할·통합 표현이므로 하나의 CSV만 사용했습니다.")
    st.write("제공된 두 ZIP 중 전체 네 품목을 모두 포함한 kat_sale_citrus_merged.zip만 모델 입력으로 사용했습니다.")
    st.markdown(f"[정제 데이터 상세 조사 보고서 보기]({(BASE_DIR / 'data_audit.md').name})")

st.caption("미래 가격·거래량은 RandomForest로 미리 계산해 CSV에 저장한 결과입니다. 화면을 열 때 모델 학습이나 예측 계산은 실행되지 않습니다.")
