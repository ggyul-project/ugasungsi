import os
import glob
import math
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = Path("../data")
PKL_PATH = "streamlit_predictions.pkl"

TARGET_VARIETIES = ["조생귤", "한라봉", "하우스감귤", "천혜향"]

MARKET_SPECS = [
    ("서울가락시장", "서울가락"),
    ("대구북부시장", "대구북부"),
    ("부산반여시장", "부산반여"),
    ("경기구리시장", "구리"),
    ("부산엄궁시장", "부산엄궁"),
]

SEOLLAL_DATES = pd.to_datetime([
    "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
    "2025-01-29", "2026-02-17", "2027-02-07",
])
CHUSEOK_DATES = pd.to_datetime([
    "2021-09-21", "2022-09-10", "2023-09-29", "2024-09-17",
    "2025-10-06", "2026-09-25", "2027-09-15",
])

CITRUS_VARIETY_MAP = {
    1: "조생귤", "1": "조생귤", "01": "조생귤",
    2: "만생귤", "2": "만생귤", "02": "만생귤",
    3: "극조생감귤", "3": "극조생감귤", "03": "극조생감귤",
    4: "하우스감귤", "4": "하우스감귤", "04": "하우스감귤",
    5: "비가림감귤", "5": "비가림감귤", "05": "비가림감귤",
    6: "월동감귤", "6": "월동감귤", "06": "월동감귤",
    7: "한라봉", "7": "한라봉", "07": "한라봉",
    8: "천혜향", "8": "천혜향", "08": "천혜향",
    9: "레드향", "9": "레드향", "09": "레드향",
    10: "카라향", "10": "카라향",
    11: "황금향", "11": "황금향",
    12: "하루미", "12": "하루미",
    13: "온주", "13": "온주",
    15: "성전", "15": "성전",
    17: "홍미향", "17": "홍미향",
    19: "타이벡감귤", "19": "타이벡감귤",
    20: "풋귤(청귤)", "20": "풋귤(청귤)",
    23: "미니향", "23": "미니향",
    32: "유라와세", "32": "유라와세",
    34: "산하홍", "34": "산하홍",
    98: "감귤(수입)", "98": "감귤(수입)",
    99: "기타", "99": "기타",
}

USER_OVERRIDE_MAP = {
    7: "한라봉", "7": "한라봉", "07": "한라봉", 7.0: "한라봉",
    8: "천혜향", "8": "천혜향", "08": "천혜향", 8.0: "천혜향",
    9: "레드향", "9": "레드향", "09": "레드향", 9.0: "레드향",
    10: "카라향", "10": "카라향", 10.0: "카라향",
    11: "황금향", "11": "황금향", 11.0: "황금향",
}

FEATURE_COLS_VOL = [
    "log_grp_vol_mean", "norm_unit_weight", "seasonal_log_vol",
    "seollal_pre_surge", "seollal_post_dip", "chuseok_pre_surge", "chuseok_post_dip",
    "holiday_vol_effect",
    "wx_temp_wk1_norm", "wx_temp_anom_norm", "wx_rain_wk1_norm", "wx_sun_wk1_norm",
    "wx_snow_wk1_norm", "wx_vol_effect",
    "sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2", "sin_dow", "cos_dow", "year_norm",
    "log_vol_lag_1", "log_vol_lag_2", "log_vol_lag_3", "log_vol_lag_7", "vol_diff_1w",
    "yoy_same_period_log_vol", "yoy_same_period_vol_diff", "vol_yoy_change",
    "vol_roll7_mean", "vol_roll14_mean", "norm_prc_lag_1", "ratio_teuk_lag1",
]

FEATURE_COLS_PRC = [
    "norm_grp_prc_mean", "norm_unit_weight", "norm_seasonal_prc",
    "seollal_pre_surge", "seollal_post_dip", "chuseok_pre_surge", "chuseok_post_dip",
    "holiday_prc_effect",
    "wx_temp_wk1_norm", "wx_temp_anom_norm", "wx_rain_wk1_norm", "wx_sun_wk1_norm",
    "wx_snow_wk1_norm", "wx_prc_effect",
    "sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2", "sin_dow", "cos_dow", "year_norm",
    "norm_prc_lag_1", "norm_prc_lag_2", "norm_prc_lag_3", "norm_prc_lag_7", "prc_diff_1w",
    "yoy_same_period_norm_prc", "yoy_same_period_prc_diff", "prc_yoy_change",
    "norm_prc_roll7", "norm_prc_roll14", "log_vol_lag_1", "ratio_teuk_lag1",
]

LAG_COLS_VOL = [
    "log_vol_lag_1", "log_vol_lag_2", "log_vol_lag_3",
    "log_vol_lag_7", "log_vol_lag_14", "log_vol_lag_30",
    "yoy_same_period_log_vol", "exp_vol_yoy_chg",
]

LAG_COLS_PRC = [
    "norm_prc_lag_1", "norm_prc_lag_2", "norm_prc_lag_3",
    "norm_prc_lag_7", "norm_prc_lag_14", "norm_prc_lag_30",
    "yoy_same_period_norm_prc", "exp_prc_yoy_chg",
]

WX_MERGE_COLS = [
    "date", "wx_temp_lag7", "wx_temp_anom_lag7", "wx_rain_dur_lag7",
    "wx_sunshine_lag7", "wx_snow_lag7",
    "wx_temp_wk1_mean", "wx_temp_anom_wk1", "wx_rain_dur_wk1_sum",
    "wx_rain_max_wk1", "wx_sunshine_wk1_mean", "wx_snow_wk1_max",
    "wx_temp_wk1_norm", "wx_temp_anom_norm", "wx_rain_wk1_norm",
    "wx_sun_wk1_norm", "wx_snow_wk1_norm",
]


def build_korean_holidays():
    fixed_mmdd = ["01-01", "03-01", "05-05", "06-06", "08-15", "10-03", "10-09", "12-25"]
    holidays = set()
    for y in range(2021, 2028):
        for mmdd in fixed_mmdd:
            holidays.add(pd.Timestamp(f"{y}-{mmdd}"))
    for s_dt in SEOLLAL_DATES:
        for delta in [-1, 0, 1, 2]:
            holidays.add(s_dt + pd.Timedelta(days=delta))
    for c_dt in CHUSEOK_DATES:
        for delta in [-1, 0, 1, 2]:
            holidays.add(c_dt + pd.Timedelta(days=delta))
    return holidays


def load_weather():
    fc_raw = pd.read_csv(DATA_DIR / "forecast.csv", encoding="cp949")
    fc_raw = fc_raw.rename(columns={
        "지점": "station_id",
        "지점명": "station_name",
        "일시": "date",
        "평균기온(°C)": "temp_avg",
        "강수 계속시간(hr)": "rain_dur_hr",
        "10분 최다 강수량(mm)": "rain_10m_max",
        "합계 일조시간(hr)": "sunshine_hr",
        "일 최심신적설(cm)": "snow_depth_cm",
    })
    fc_raw["date"] = pd.to_datetime(fc_raw["date"], errors="coerce")
    for wcol in ["rain_dur_hr", "rain_10m_max", "sunshine_hr", "snow_depth_cm"]:
        fc_raw[wcol] = pd.to_numeric(fc_raw[wcol], errors="coerce").fillna(0.0)
    fc_raw["temp_avg"] = pd.to_numeric(fc_raw["temp_avg"], errors="coerce")
    fc_raw["temp_avg"] = fc_raw.groupby("station_name")["temp_avg"].transform(
        lambda s: s.interpolate().bfill().ffill()
    )

    daily_wx = (
        fc_raw.groupby("date", as_index=False)
        .agg(
            temp_avg=("temp_avg", "mean"),
            rain_dur_hr=("rain_dur_hr", "mean"),
            rain_10m_max=("rain_10m_max", "max"),
            sunshine_hr=("sunshine_hr", "mean"),
            snow_depth_cm=("snow_depth_cm", "max"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    full_wx_dates = pd.date_range(daily_wx["date"].min(), "2027-02-28", freq="D")
    daily_wx = pd.DataFrame({"date": full_wx_dates}).merge(daily_wx, on="date", how="left")
    daily_wx["doy"] = daily_wx["date"].dt.dayofyear
    doy_clim = daily_wx.groupby("doy")[
        ["temp_avg", "rain_dur_hr", "rain_10m_max", "sunshine_hr", "snow_depth_cm"]
    ].transform("mean")
    for wcol in ["temp_avg", "rain_dur_hr", "rain_10m_max", "sunshine_hr", "snow_depth_cm"]:
        daily_wx[wcol] = daily_wx[wcol].fillna(doy_clim[wcol])

    daily_wx["temp_anom"] = daily_wx["temp_avg"] - doy_clim["temp_avg"]
    daily_wx["wx_temp_lag7"] = daily_wx["temp_avg"].shift(7).bfill()
    daily_wx["wx_temp_anom_lag7"] = daily_wx["temp_anom"].shift(7).fillna(0.0)
    daily_wx["wx_rain_dur_lag7"] = daily_wx["rain_dur_hr"].shift(7).fillna(0.0)
    daily_wx["wx_rain_max_lag7"] = daily_wx["rain_10m_max"].shift(7).fillna(0.0)
    daily_wx["wx_sunshine_lag7"] = daily_wx["sunshine_hr"].shift(7).bfill()
    daily_wx["wx_snow_lag7"] = daily_wx["snow_depth_cm"].shift(7).fillna(0.0)

    daily_wx["wx_temp_wk1_mean"] = daily_wx["temp_avg"].shift(1).rolling(7, min_periods=1).mean().bfill()
    daily_wx["wx_temp_anom_wk1"] = daily_wx["temp_anom"].shift(1).rolling(7, min_periods=1).mean().fillna(0.0)
    daily_wx["wx_rain_dur_wk1_sum"] = daily_wx["rain_dur_hr"].shift(1).rolling(7, min_periods=1).sum().fillna(0.0)
    daily_wx["wx_rain_max_wk1"] = daily_wx["rain_10m_max"].shift(1).rolling(7, min_periods=1).max().fillna(0.0)
    daily_wx["wx_sunshine_wk1_mean"] = daily_wx["sunshine_hr"].shift(1).rolling(7, min_periods=1).mean().bfill()
    daily_wx["wx_snow_wk1_max"] = daily_wx["snow_depth_cm"].shift(1).rolling(7, min_periods=1).max().fillna(0.0)

    daily_wx["wx_temp_wk1_norm"] = (daily_wx["wx_temp_wk1_mean"] - 16.0) / 8.0
    daily_wx["wx_temp_anom_norm"] = (daily_wx["wx_temp_anom_wk1"] / 3.0).clip(-3.0, 3.0)
    daily_wx["wx_rain_wk1_norm"] = np.log1p(daily_wx["wx_rain_dur_wk1_sum"]) / 3.0
    daily_wx["wx_sun_wk1_norm"] = (daily_wx["wx_sunshine_wk1_mean"] - 5.5) / 3.0
    daily_wx["wx_snow_wk1_norm"] = np.log1p(daily_wx["wx_snow_wk1_max"])
    return daily_wx


def compute_holiday_features(date_series):
    unique_dates = pd.Index(pd.to_datetime(date_series).unique()).sort_values()
    h_rows = []
    for d in unique_dates:
        d_seol = min([(d - s).days for s in SEOLLAL_DATES], key=lambda x: abs(x))
        d_chu = min([(d - c).days for c in CHUSEOK_DATES], key=lambda x: abs(x))
        seol_pre = math.exp(-((d_seol + 5.5) ** 2) / (2 * 4.5 ** 2)) if -18 <= d_seol <= -1 else 0.0
        seol_post = math.exp(-((d_seol - 4.0) ** 2) / (2 * 3.5 ** 2)) if 1 <= d_seol <= 12 else 0.0
        chu_pre = math.exp(-((d_chu + 5.5) ** 2) / (2 * 4.5 ** 2)) if -18 <= d_chu <= -1 else 0.0
        chu_post = math.exp(-((d_chu - 4.0) ** 2) / (2 * 3.5 ** 2)) if 1 <= d_chu <= 12 else 0.0

        if -14 <= d_seol <= -1:
            h_period = "설날 전(D-14~D-1)"
        elif 1 <= d_seol <= 7:
            h_period = "설날 후(D+1~D+7)"
        elif -14 <= d_chu <= -1:
            h_period = "추석 전(D-14~D-1)"
        elif 1 <= d_chu <= 7:
            h_period = "추석 후(D+1~D+7)"
        else:
            h_period = "평시"

        h_rows.append({
            "date": d,
            "d_seol": d_seol,
            "d_chu": d_chu,
            "seollal_pre_surge": seol_pre,
            "seollal_post_dip": seol_post,
            "chuseok_pre_surge": chu_pre,
            "chuseok_post_dip": chu_post,
            "holiday_period": h_period,
        })
    return pd.DataFrame(h_rows)


def format_week_intervals(weeks):
    if not weeks:
        return "없음"
    weeks = sorted(weeks)
    ranges = []
    s, p = weeks[0], weeks[0]
    for w in weeks[1:]:
        if w == p + 1:
            p = w
        else:
            ranges.append((s, p))
            s, p = w, w
    ranges.append((s, p))
    out = []
    for s_w, e_w in ranges:
        d1 = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=(s_w - 1) * 7)).strftime("%m/%d")
        d2 = min(
            pd.Timestamp("2025-12-31"),
            pd.Timestamp("2025-01-01") + pd.Timedelta(days=e_w * 7 - 1),
        ).strftime("%m/%d")
        out.append(f"{s_w}~{e_w}주차({d1}~{d2})" if s_w != e_w else f"{s_w}주차({d1}~{d2})")
    return ", ".join(out)


def map_grade_5(g):
    if g in ("특", "특품"):
        return "특"
    elif g in ("상", "상품"):
        return "상"
    elif g in ("중", "보통", "중품"):
        return "중"
    elif g in ("하", "하품"):
        return "하"
    return "기타"


def load_all_raw_csvs():
    t0 = time.time()
    data_files = sorted(list(set(
        glob.glob(str(DATA_DIR / "kat*.csv"))
        + glob.glob(str(DATA_DIR / "mkat*.csv"))
        + glob.glob(str(DATA_DIR / "*kat*.csv"))
    )))
    valid_files = [f for f in data_files if "_all.csv" not in f and os.path.getsize(f) > 50]
    target_cols = [
        "trd_clcln_ymd", "whsl_mrkt_cd", "whsl_mrkt_nm", "trd_se",
        "gds_sclsf_cd", "gds_sclsf_nm", "grd_cd", "grd_nm",
        "pkg_cd", "pkg_nm", "unit_nm", "unit_qty", "unit_tot_qty", "avgprc", "totprc",
    ]
    dfs = [
        pd.read_csv(f, usecols=lambda c: c in target_cols, low_memory=False, encoding="utf-8-sig")
        for f in valid_files
    ]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()

    target_raw_mkts = [raw_nm for _, raw_nm in MARKET_SPECS]
    df = df[df["whsl_mrkt_nm"].isin(target_raw_mkts)].copy()

    df["date"] = pd.to_datetime(df["trd_clcln_ymd"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df[(df["unit_tot_qty"] > 0) & (df["totprc"] > 0) & (df["avgprc"] > 0)].copy()

    df["variety"] = (
        df["gds_sclsf_nm"]
        .combine_first(df["gds_sclsf_cd"].map(USER_OVERRIDE_MAP))
        .combine_first(df["gds_sclsf_cd"].map(CITRUS_VARIETY_MAP))
        .fillna("기타")
    )
    df = df[df["variety"].isin(TARGET_VARIETIES)].copy()

    df["is_auction"] = df["trd_se"].fillna("").astype(str).str.contains("경매", na=False).astype(float)
    df["grade_5"] = df["grd_nm"].fillna(".").astype(str).str.strip().apply(map_grade_5)
    df["volume_kg"] = pd.to_numeric(df["unit_tot_qty"], errors="coerce").fillna(0.0)
    df["unit_weight"] = pd.to_numeric(df["unit_qty"], errors="coerce").fillna(0.0)
    df["total_price"] = pd.to_numeric(df["totprc"], errors="coerce").fillna(0.0)
    df["box_price"] = pd.to_numeric(df["avgprc"], errors="coerce").fillna(0.0)
    df["unit_price_kg"] = df["total_price"] / (df["volume_kg"] + 1e-8)

    df["box_weight"] = df["unit_weight"].apply(lambda w: f"{w:g}kg" if pd.notnull(w) and w > 0 else "기타")
    df["pkg_clean"] = df["pkg_nm"].replace({".": "상자", "": "상자", np.nan: "상자"}).fillna("상자")
    df["box_unit"] = (df["box_weight"] + df["pkg_clean"]).replace({
        "3kg파렛트": "3kg상자",
        "5kg파렛트": "5kg상자",
        "9kg파렛트": "9kg상자",
        "10kg파렛트": "10kg상자",
    })
    df = df[(df["unit_price_kg"] >= 100) & (df["unit_price_kg"] <= 100000)].copy()
    print(f"Loaded {len(valid_files)} CSVs -> 5 markets & 4 varieties: {len(df):,} rows ({time.time() - t0:.1f}s)", flush=True)
    return df


class GatedResidualNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.05):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(out_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        h = self.drop(self.fc2(self.elu(self.fc1(x))))
        g = torch.sigmoid(self.gate(h))
        return self.norm(self.skip(x) + h * g)


class TFTQuantileNet(nn.Module):
    def __init__(self, n_features, n_lags=8, d_model=64, dropout=0.05, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles
        self.vsn = nn.Sequential(nn.Linear(n_features, n_features), nn.Softmax(dim=-1))
        self.grn_feat = GatedResidualNetwork(n_features, d_model, d_model, dropout)
        self.grn_lag = GatedResidualNetwork(n_lags, d_model // 2, d_model // 2, dropout)
        self.grn_out = GatedResidualNetwork(d_model + d_model // 2, d_model, d_model, dropout)
        self.head = nn.Linear(d_model, len(quantiles))

    def forward(self, x, l):
        w = self.vsn(x)
        h_f = self.grn_feat(x * w * x.shape[1])
        h_l = self.grn_lag(l)
        h = self.grn_out(torch.cat([h_f, h_l], dim=-1))
        return self.head(h)


def train_tft_model(X_tr, L_tr, y_tr, X_va, L_va, y_va, epochs=55, lr=0.005, batch_size=2048, seed=42):
    torch.manual_seed(seed)
    model = TFTQuantileNet(X_tr.shape[1], L_tr.shape[1], d_model=64, dropout=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.15)
    qs = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32).view(1, 3)

    x_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    l_tr_t = torch.tensor(L_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)

    x_va_t = torch.tensor(X_va, dtype=torch.float32)
    l_va_t = torch.tensor(L_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32).view(-1, 1)

    best_loss = float("inf")
    best_state = None
    n = len(X_tr)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x_tr_t[idx], l_tr_t[idx])
            err = y_tr_t[idx] - pred
            loss = torch.maximum(qs * err, (qs - 1.0) * err).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        if len(X_va) > 0 and ep >= int(epochs * 0.6):
            model.eval()
            with torch.no_grad():
                vpred = model(x_va_t, l_va_t)
                verr = y_va_t - vpred
                vloss = float(torch.maximum(qs * verr, (qs - 1.0) * verr).mean().item())
            if vloss < best_loss:
                best_loss = vloss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_tft(model, X, L):
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32), torch.tensor(L, dtype=torch.float32)).numpy()
    return np.sort(out, axis=1)


def train_xgb_quantiles(XL_tr, y_tr, XL_va, y_va, n_estimators=260, max_depth=6, lr=0.045, seed=42):
    models = {}
    for q in (0.1, 0.5, 0.9):
        m = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=1.5,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        if len(XL_va) > 0:
            m.fit(XL_tr, y_tr, eval_set=[(XL_va, y_va)], verbose=False)
        else:
            m.fit(XL_tr, y_tr, verbose=False)
        models[q] = m
    return models


def predict_quantile_dict(models, XL):
    out = np.column_stack([models[q].predict(XL) for q in (0.1, 0.5, 0.9)])
    return np.sort(out, axis=1)


def train_lgb_quantiles(XL_tr, y_tr, XL_va, y_va, n_estimators=280, num_leaves=63, max_depth=9, lr=0.045, seed=42):
    models = {}
    for q in (0.1, 0.5, 0.9):
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=n_estimators,
            learning_rate=lr,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=15,
            subsample=0.88,
            subsample_freq=1,
            colsample_bytree=0.88,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
        m.fit(XL_tr, y_tr)
        models[q] = m
    return models


def train_rf_quantile(XL_tr, y_tr, n_estimators=160, max_depth=18, seed=42):
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features=0.76,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(XL_tr, y_tr)
    return rf


def predict_rf_quantile(rf, XL):
    tree_preds = np.array([est.predict(XL) for est in rf.estimators_])
    q10 = np.percentile(tree_preds, 10, axis=0)
    q50 = np.mean(tree_preds, axis=0)
    q90 = np.percentile(tree_preds, 90, axis=0)
    return np.sort(np.column_stack([q10, q50, q90]), axis=1)


def prep_arrays(df_part, f_cols, l_cols, mean_vec=None, std_vec=None):
    X = df_part[f_cols].fillna(0.0).values.astype(np.float32)
    L = df_part[l_cols].fillna(0.0).values.astype(np.float32)
    if mean_vec is None:
        mean_vec = X.mean(axis=0)
        std_vec = X.std(axis=0)
        std_vec = np.where(std_vec < 1e-6, 1.0, std_vec)
    X_norm = np.clip((X - mean_vec) / std_vec, -8.0, 8.0)
    return X_norm, L, mean_vec, std_vec


def build_single_market_bundle(df_mkt, mkt_disp_name, daily_wx, korean_holidays):
    t_m = time.time()
    df = df_mkt.copy()

    var_order_all = df.groupby("variety")["volume_kg"].sum().sort_values(ascending=False).index.tolist()
    variety_order = [v for v in var_order_all if v in TARGET_VARIETIES]

    unit_stats = (
        df.groupby(["variety", "box_unit"], as_index=False)
        .agg(
            거래건수=("date", "count"),
            총거래량_kg=("volume_kg", "sum"),
            총거래금액_원=("total_price", "sum"),
            평균_상자당가격=("box_price", "mean"),
            단위중량_kg=("unit_weight", "median"),
        )
    )
    var_tot = df.groupby("variety", as_index=False).agg(
        품종총건수=("date", "count"),
        품종총물량_kg=("volume_kg", "sum"),
    )
    unit_stats = unit_stats.merge(var_tot, on="variety", how="left")
    unit_stats["품종내_건수비중(%)"] = (unit_stats["거래건수"] / unit_stats["품종총건수"]) * 100.0
    unit_stats["품종내_물량비중(%)"] = (unit_stats["총거래량_kg"] / unit_stats["품종총물량_kg"]) * 100.0
    unit_stats["거래량_톤"] = unit_stats["총거래량_kg"] / 1000.0
    unit_stats["가중평균_kg단가"] = unit_stats["총거래금액_원"] / (unit_stats["총거래량_kg"] + 1e-8)

    selected_parts = []
    for v in variety_order:
        sub_v = unit_stats[unit_stats["variety"] == v]
        filt = sub_v[(sub_v["거래건수"] >= 100) & (sub_v["품종내_건수비중(%)"] >= 1.0)]
        if len(filt) == 0:
            filt = sub_v[(sub_v["거래건수"] >= 30) & (sub_v["품종내_건수비중(%)"] >= 1.0)]
        if len(filt) == 0:
            filt = sub_v.nlargest(3, "거래건수")
        selected_parts.append(filt.sort_values("거래건수", ascending=False).head(5))

    top5_units = pd.concat(selected_parts, ignore_index=True)
    top5_units["variety_rank"] = top5_units["variety"].map({v: i for i, v in enumerate(variety_order)})
    top5_units = top5_units.sort_values(["variety_rank", "거래건수"], ascending=[True, False]).reset_index(drop=True)
    top5_units["품종내_순위"] = top5_units.groupby("variety").cumcount() + 1
    top5_units["group_id"] = top5_units["variety"] + "_" + top5_units["box_unit"]

    daily_mkt_tot = df.groupby("date")["volume_kg"].sum()
    active_dates = set(daily_mkt_tot[daily_mkt_tot >= 100.0].index)
    df = df[(df["date"].dt.dayofweek != 6) & (df["date"].isin(active_dates))].copy()

    df["week_idx"] = np.minimum(52, (df["date"].dt.dayofyear - 1) // 7 + 1)
    wk_raw = df.groupby(["variety", "year", "week_idx"], as_index=False).agg(
        vol_ton=("volume_kg", lambda s: s.sum() / 1000.0),
        tx_cnt=("date", "count"),
        n_days=("date", "nunique"),
    )
    wk_raw["daily_vol_ton"] = wk_raw["vol_ton"] / np.maximum(wk_raw["n_days"], 1)

    all_yrs = sorted(df["year"].unique())
    wk_grid = (
        pd.MultiIndex.from_product([variety_order, all_yrs, range(1, 53)], names=["variety", "year", "week_idx"])
        .to_frame()
        .reset_index(drop=True)
        .merge(wk_raw, on=["variety", "year", "week_idx"], how="left")
    )
    for col in ["vol_ton", "tx_cnt", "n_days", "daily_vol_ton"]:
        wk_grid[col] = wk_grid[col].fillna(0.0)

    var_yr_totals = wk_grid.groupby(["variety", "year"], as_index=False)["vol_ton"].sum()
    valid_var_yrs = var_yr_totals[var_yr_totals["vol_ton"] > 0.1][["variety", "year"]]
    wk_grid_valid = wk_grid.merge(valid_var_yrs, on=["variety", "year"], how="inner")

    wk_stats = wk_grid_valid.groupby(["variety", "week_idx"], as_index=False).agg(
        mean_wk_ton=("vol_ton", "mean"),
        mean_daily_ton=("daily_vol_ton", "mean"),
        mean_tx_cnt=("tx_cnt", "mean"),
    )
    var_peak = wk_stats.groupby("variety", as_index=False).agg(
        peak_wk_ton=("mean_wk_ton", "max"),
        annual_wk_sum=("mean_wk_ton", "sum"),
    )
    wk_stats = wk_stats.merge(var_peak, on="variety", how="left")
    wk_stats["pct_of_peak"] = wk_stats["mean_wk_ton"] / (wk_stats["peak_wk_ton"] + 1e-8) * 100.0
    wk_stats["pct_of_annual"] = wk_stats["mean_wk_ton"] / (wk_stats["annual_wk_sum"] + 1e-8) * 100.0

    wk_stats["is_off_raw"] = (
        (wk_stats["pct_of_peak"] < 1.5)
        | (wk_stats["pct_of_annual"] < 0.25)
        | ((wk_stats["mean_wk_ton"] < 0.05) & (wk_stats["mean_tx_cnt"] < 2.0))
    )
    wk_stats["is_offseason"] = False
    for v in variety_order:
        sub_w = wk_stats[wk_stats["variety"] == v].sort_values("week_idx")
        act = (~sub_w["is_off_raw"].values).astype(int)
        act_pad = np.concatenate([[act[-1]], act, [act[0]]])
        for i in range(1, 53):
            if act_pad[i] == 1 and act_pad[i - 1] == 0 and act_pad[i + 1] == 0:
                act_pad[i] = 0
        act = act_pad[1:53]
        act_pad = np.concatenate([act[-2:], act, act[:2]])
        for i in range(2, 54):
            if act_pad[i] == 0 and act_pad[i - 1] == 1 and act_pad[i + 1] == 1:
                act_pad[i] = 1
            elif act_pad[i] == 0 and act_pad[i + 1] == 0 and act_pad[i - 1] == 1 and act_pad[i + 2] == 1:
                act_pad[i] = 1
                act_pad[i + 1] = 1
        act = act_pad[2:54]
        wk_stats.loc[sub_w.index, "is_offseason"] = act == 0

    active_season_pairs = set(
        zip(
            wk_stats[~wk_stats["is_offseason"]]["variety"],
            wk_stats[~wk_stats["is_offseason"]]["week_idx"],
        )
    )

    offseason_rows = []
    for v in variety_order:
        sub_w = wk_stats[wk_stats["variety"] == v].sort_values("week_idx")
        act_w = sub_w[~sub_w["is_offseason"]]["week_idx"].tolist()
        off_w = sub_w[sub_w["is_offseason"]]["week_idx"].tolist()
        act_share = sub_w[~sub_w["is_offseason"]]["mean_wk_ton"].sum() / (sub_w["mean_wk_ton"].sum() + 1e-8) * 100.0
        off_mean_ton = sub_w[sub_w["is_offseason"]]["mean_wk_ton"].mean() if len(off_w) > 0 else 0.0
        act_mean_ton = sub_w[~sub_w["is_offseason"]]["mean_wk_ton"].mean() if len(act_w) > 0 else 0.0
        offseason_rows.append({
            "품종명": v,
            "출하기 주수": f"{len(act_w)}주 / 52주",
            "비수기(예측제외) 주수": f"{len(off_w)}주 / 52주",
            "출하기 물량비중": f"{act_share:.1f}%",
            "출하기 주평균 물량": f"{act_mean_ton:,.1f} 톤/주",
            "비수기 주평균 물량": f"{off_mean_ton:,.2f} 톤/주",
            "출하기(예측 수행 기간)": format_week_intervals(act_w),
            "계절적 비수기(예측 제외 기간)": format_week_intervals(off_w),
        })
    offseason_summary_df = pd.DataFrame(offseason_rows)

    df_target = df.merge(
        top5_units[["variety", "box_unit", "group_id", "단위중량_kg"]],
        on=["variety", "box_unit"],
        how="inner",
    )
    df_target["vol_teuk"] = np.where(df_target["grade_5"] == "특", df_target["volume_kg"], 0.0)

    daily_df = (
        df_target.groupby(["group_id", "variety", "box_unit", "단위중량_kg", "date"], as_index=False)
        .agg(
            tx_count=("trd_clcln_ymd", "count"),
            volume_kg=("volume_kg", "sum"),
            total_price=("total_price", "sum"),
            box_price_mean=("box_price", "mean"),
            auction_ratio=("is_auction", "mean"),
            vol_teuk=("vol_teuk", "sum"),
        )
    )
    daily_df["week_idx"] = np.minimum(52, (daily_df["date"].dt.dayofyear - 1) // 7 + 1)
    daily_df = daily_df[
        [(v, w) in active_season_pairs for v, w in zip(daily_df["variety"], daily_df["week_idx"])]
    ].sort_values(["group_id", "date"]).reset_index(drop=True)

    daily_df["unit_weight_kg"] = np.where(daily_df["단위중량_kg"] > 0, daily_df["단위중량_kg"], 5.0)
    daily_df["volume_ton"] = daily_df["volume_kg"] / 1000.0
    daily_df["unit_price_kg"] = daily_df["total_price"] / (daily_df["volume_kg"] + 1e-8)
    daily_df["ratio_teuk"] = daily_df["vol_teuk"] / (daily_df["volume_kg"] + 1e-8)
    daily_df["year"] = daily_df["date"].dt.year
    daily_df["month"] = daily_df["date"].dt.month
    daily_df["dayofyear"] = daily_df["date"].dt.dayofyear
    daily_df["dayofweek"] = daily_df["date"].dt.dayofweek

    hol_df = compute_holiday_features(daily_df["date"])
    daily_df = daily_df.merge(hol_df, on="date", how="left")
    daily_df = daily_df.merge(daily_wx[WX_MERGE_COLS], on="date", how="left")
    daily_df["wx_rain_regime"] = pd.cut(
        daily_df["wx_rain_dur_wk1_sum"],
        bins=[-1, 3.0, 15.0, 500.0],
        labels=["건조·청명(0~3h)", "보통강우(3~15h)", "장시간강우(15h+)"],
    ).astype(str)

    tr_mask = daily_df["year"] <= 2025

    grp_prof = (
        daily_df[tr_mask]
        .groupby("group_id", as_index=False)
        .agg(
            grp_vol_mean=("volume_ton", "mean"),
            grp_prc_mean=("unit_price_kg", "median"),
            grp_prc_std=("unit_price_kg", "std"),
        )
    )
    missing_gids = set(daily_df["group_id"].unique()) - set(grp_prof["group_id"].unique())
    if missing_gids:
        extra_prof = (
            daily_df[daily_df["group_id"].isin(missing_gids)]
            .groupby("group_id", as_index=False)
            .agg(
                grp_vol_mean=("volume_ton", "mean"),
                grp_prc_mean=("unit_price_kg", "median"),
                grp_prc_std=("unit_price_kg", "std"),
            )
        )
        grp_prof = pd.concat([grp_prof, extra_prof], ignore_index=True)
    grp_prof["grp_prc_std"] = grp_prof["grp_prc_std"].fillna(400.0).clip(lower=150.0, upper=1500.0)

    grp_month_prof = (
        daily_df[tr_mask]
        .groupby(["group_id", "month"], as_index=False)
        .agg(seasonal_vol_mean=("volume_ton", "mean"), seasonal_prc_mean=("unit_price_kg", "median"))
    )
    grp_hol_prof = (
        daily_df[tr_mask]
        .groupby(["group_id", "holiday_period"], as_index=False)
        .agg(hol_prc_median=("unit_price_kg", "median"), hol_vol_mean=("volume_ton", "mean"))
    )
    grp_wx_prof = (
        daily_df[tr_mask]
        .groupby(["group_id", "month", "wx_rain_regime"], as_index=False)
        .agg(wx_prc_median=("unit_price_kg", "median"), wx_vol_mean=("volume_ton", "mean"))
    )

    daily_df = daily_df.merge(grp_prof, on="group_id", how="left")
    daily_df = daily_df.merge(grp_month_prof, on=["group_id", "month"], how="left")
    daily_df = daily_df.merge(grp_hol_prof, on=["group_id", "holiday_period"], how="left")
    daily_df = daily_df.merge(grp_wx_prof, on=["group_id", "month", "wx_rain_regime"], how="left")

    daily_df["seasonal_vol_mean"] = daily_df["seasonal_vol_mean"].fillna(daily_df["grp_vol_mean"])
    daily_df["seasonal_prc_mean"] = daily_df["seasonal_prc_mean"].fillna(daily_df["grp_prc_mean"])
    daily_df["hol_prc_median"] = daily_df["hol_prc_median"].fillna(daily_df["seasonal_prc_mean"])
    daily_df["hol_vol_mean"] = daily_df["hol_vol_mean"].fillna(daily_df["seasonal_vol_mean"])
    daily_df["wx_prc_median"] = daily_df["wx_prc_median"].fillna(daily_df["seasonal_prc_mean"])
    daily_df["wx_vol_mean"] = daily_df["wx_vol_mean"].fillna(daily_df["seasonal_vol_mean"])

    daily_df["holiday_prc_effect"] = (
        (daily_df["hol_prc_median"] - daily_df["seasonal_prc_mean"]) / daily_df["grp_prc_std"]
    ).clip(-1.5, 1.5)
    daily_df["holiday_vol_effect"] = (
        np.log1p(daily_df["hol_vol_mean"]) - np.log1p(daily_df["seasonal_vol_mean"])
    ).clip(-1.5, 1.5)
    daily_df["wx_prc_effect"] = (
        (daily_df["wx_prc_median"] - daily_df["seasonal_prc_mean"]) / daily_df["grp_prc_std"]
    ).clip(-1.5, 1.5)
    daily_df["wx_vol_effect"] = (
        np.log1p(daily_df["wx_vol_mean"]) - np.log1p(daily_df["seasonal_vol_mean"])
    ).clip(-1.5, 1.5)

    daily_df["unit_price_kg_clean"] = np.clip(
        daily_df["unit_price_kg"],
        daily_df["grp_prc_mean"] - 2.5 * daily_df["grp_prc_std"],
        daily_df["grp_prc_mean"] + 2.5 * daily_df["grp_prc_std"],
    ).clip(lower=400.0)

    daily_df["sin_doy1"] = np.sin(2 * np.pi * daily_df["dayofyear"] / 365.25)
    daily_df["cos_doy1"] = np.cos(2 * np.pi * daily_df["dayofyear"] / 365.25)
    daily_df["sin_doy2"] = np.sin(4 * np.pi * daily_df["dayofyear"] / 365.25)
    daily_df["cos_doy2"] = np.cos(4 * np.pi * daily_df["dayofyear"] / 365.25)
    daily_df["sin_dow"] = np.sin(2 * np.pi * daily_df["dayofweek"] / 6.0)
    daily_df["cos_dow"] = np.cos(2 * np.pi * daily_df["dayofweek"] / 6.0)
    daily_df["year_norm"] = (daily_df["year"] - 2021) / 5.0

    daily_df["log_vol"] = np.log1p(daily_df["volume_ton"])
    daily_df["seasonal_log_vol"] = np.log1p(daily_df["seasonal_vol_mean"])
    daily_df["log_grp_vol_mean"] = np.log1p(daily_df["grp_vol_mean"])
    daily_df["norm_grp_prc_mean"] = daily_df["grp_prc_mean"] / 5000.0
    daily_df["norm_unit_weight"] = daily_df["unit_weight_kg"] / 10.0
    daily_df["norm_seasonal_prc"] = (daily_df["seasonal_prc_mean"] - daily_df["grp_prc_mean"]) / daily_df["grp_prc_std"]
    daily_df["norm_prc_clean"] = (daily_df["unit_price_kg_clean"] - daily_df["grp_prc_mean"]) / daily_df["grp_prc_std"]

    for lag in [1, 2, 3, 7, 14, 30]:
        daily_df[f"log_vol_lag_{lag}"] = (
            daily_df.groupby("group_id")["log_vol"].shift(lag).fillna(daily_df["seasonal_log_vol"])
        )
        prc_s = daily_df.groupby("group_id")["unit_price_kg_clean"].shift(lag).fillna(daily_df["seasonal_prc_mean"])
        daily_df[f"norm_prc_lag_{lag}"] = (prc_s - daily_df["grp_prc_mean"]) / daily_df["grp_prc_std"]

    daily_df["vol_roll7_mean"] = (
        daily_df.groupby("group_id")["log_vol"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        .fillna(daily_df["seasonal_log_vol"])
    )
    daily_df["vol_roll14_mean"] = (
        daily_df.groupby("group_id")["log_vol"]
        .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        .fillna(daily_df["seasonal_log_vol"])
    )
    daily_df["norm_prc_roll7"] = (
        daily_df.groupby("group_id")["norm_prc_clean"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        .fillna(daily_df["norm_seasonal_prc"])
    )
    daily_df["norm_prc_roll14"] = (
        daily_df.groupby("group_id")["norm_prc_clean"]
        .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        .fillna(daily_df["norm_seasonal_prc"])
    )
    daily_df["ratio_teuk_lag1"] = daily_df.groupby("group_id")["ratio_teuk"].shift(1).fillna(0.5)

    hist_wk_prof = (
        daily_df[tr_mask]
        .groupby(["group_id", "week_idx"], as_index=False)
        .agg(
            hist_wk_vol_ton=("volume_ton", "mean"),
            hist_wk_log_vol=("log_vol", "mean"),
            hist_wk_prc_kg=("unit_price_kg_clean", "median"),
            hist_wk_norm_prc=("norm_prc_clean", "mean"),
        )
    )
    gids = sorted(top5_units["group_id"].unique().tolist())
    wk_full_grid = (
        pd.MultiIndex.from_product([gids, range(1, 53)], names=["group_id", "week_idx"])
        .to_frame()
        .reset_index(drop=True)
        .merge(grp_prof, on="group_id", how="left")
    )
    wk_full_grid["log_grp_vol_mean"] = np.log1p(wk_full_grid["grp_vol_mean"])
    wk_full_grid = wk_full_grid.merge(hist_wk_prof, on=["group_id", "week_idx"], how="left")
    for col, fallback_col in [
        ("hist_wk_vol_ton", "grp_vol_mean"),
        ("hist_wk_log_vol", "log_grp_vol_mean"),
        ("hist_wk_prc_kg", "grp_prc_mean"),
    ]:
        wk_full_grid[col] = (
            wk_full_grid.groupby("group_id")[col]
            .transform(lambda s: s.interpolate().bfill().ffill())
            .fillna(wk_full_grid[fallback_col])
        )
    wk_full_grid["hist_wk_norm_prc"] = (
        wk_full_grid.groupby("group_id")["hist_wk_norm_prc"]
        .transform(lambda s: s.interpolate().bfill().ffill())
        .fillna(0.0)
    )
    wk_full_grid["prev_week_idx"] = np.where(wk_full_grid["week_idx"] == 1, 52, wk_full_grid["week_idx"] - 1)
    wk_full_grid = wk_full_grid.merge(
        wk_full_grid[
            ["group_id", "week_idx", "hist_wk_log_vol", "hist_wk_norm_prc", "hist_wk_vol_ton", "hist_wk_prc_kg"]
        ].rename(columns={
            "week_idx": "prev_week_idx",
            "hist_wk_log_vol": "hist_prev_wk_log_vol",
            "hist_wk_norm_prc": "hist_prev_wk_norm_prc",
            "hist_wk_vol_ton": "hist_prev_wk_vol_ton",
            "hist_wk_prc_kg": "hist_prev_wk_prc_kg",
        }),
        on=["group_id", "prev_week_idx"],
        how="left",
    )
    wk_full_grid["hist_wk_vol_diff"] = (wk_full_grid["hist_wk_log_vol"] - wk_full_grid["hist_prev_wk_log_vol"]).clip(-2.0, 2.0)
    wk_full_grid["hist_wk_prc_diff"] = (wk_full_grid["hist_wk_norm_prc"] - wk_full_grid["hist_prev_wk_norm_prc"]).clip(-2.0, 2.0)

    yr_wk_actual = (
        daily_df.groupby(["group_id", "year", "week_idx"], as_index=False)
        .agg(
            yr_wk_vol_ton=("volume_ton", "mean"),
            yr_wk_log_vol=("log_vol", "mean"),
            yr_wk_prc_kg=("unit_price_kg_clean", "median"),
            yr_wk_norm_prc=("norm_prc_clean", "mean"),
        )
    )
    yr_wk_grid = (
        pd.MultiIndex.from_product([gids, list(range(2021, 2028)), range(1, 53)], names=["group_id", "year", "week_idx"])
        .to_frame()
        .reset_index(drop=True)
        .merge(yr_wk_actual, on=["group_id", "year", "week_idx"], how="left")
        .merge(
            wk_full_grid[["group_id", "week_idx", "hist_wk_vol_ton", "hist_wk_log_vol", "hist_wk_prc_kg", "hist_wk_norm_prc", "hist_wk_vol_diff", "hist_wk_prc_diff"]],
            on=["group_id", "week_idx"],
            how="left",
        )
    )
    yr_wk_grid["yr_wk_vol_ton"] = yr_wk_grid["yr_wk_vol_ton"].fillna(yr_wk_grid["hist_wk_vol_ton"])
    yr_wk_grid["yr_wk_log_vol"] = yr_wk_grid["yr_wk_log_vol"].fillna(yr_wk_grid["hist_wk_log_vol"])
    yr_wk_grid["yr_wk_prc_kg"] = yr_wk_grid["yr_wk_prc_kg"].fillna(yr_wk_grid["hist_wk_prc_kg"])
    yr_wk_grid["yr_wk_norm_prc"] = yr_wk_grid["yr_wk_norm_prc"].fillna(yr_wk_grid["hist_wk_norm_prc"])
    yr_wk_grid = yr_wk_grid.sort_values(["group_id", "year", "week_idx"]).reset_index(drop=True)
    yr_wk_grid["yr_wk_vol_diff_1w"] = yr_wk_grid.groupby("group_id")["yr_wk_log_vol"].diff(1).fillna(yr_wk_grid["hist_wk_vol_diff"]).clip(-2.0, 2.0)
    yr_wk_grid["yr_wk_prc_diff_1w"] = yr_wk_grid.groupby("group_id")["yr_wk_norm_prc"].diff(1).fillna(yr_wk_grid["hist_wk_prc_diff"]).clip(-2.0, 2.0)

    prev_yr_lookup = yr_wk_grid[
        ["group_id", "year", "week_idx", "yr_wk_vol_ton", "yr_wk_log_vol", "yr_wk_prc_kg", "yr_wk_norm_prc", "yr_wk_vol_diff_1w", "yr_wk_prc_diff_1w"]
    ].copy()
    prev_yr_lookup["target_year"] = prev_yr_lookup["year"] + 1
    prev_yr_lookup = prev_yr_lookup.drop(columns=["year"]).rename(columns={
        "target_year": "year",
        "yr_wk_vol_ton": "prev_yr_vol_ton",
        "yr_wk_log_vol": "prev_yr_log_vol",
        "yr_wk_prc_kg": "prev_yr_prc_kg",
        "yr_wk_norm_prc": "prev_yr_norm_prc",
        "yr_wk_vol_diff_1w": "prev_yr_vol_wk_diff",
        "yr_wk_prc_diff_1w": "prev_yr_prc_wk_diff",
    })

    daily_df = daily_df.merge(
        wk_full_grid[["group_id", "week_idx", "hist_wk_vol_ton", "hist_wk_log_vol", "hist_wk_prc_kg", "hist_wk_norm_prc", "hist_wk_vol_diff", "hist_wk_prc_diff"]],
        on=["group_id", "week_idx"],
        how="left",
    )
    daily_df = daily_df.merge(prev_yr_lookup, on=["group_id", "year", "week_idx"], how="left")
    for c, fb in [
        ("prev_yr_vol_ton", "hist_wk_vol_ton"),
        ("prev_yr_log_vol", "hist_wk_log_vol"),
        ("prev_yr_prc_kg", "hist_wk_prc_kg"),
        ("prev_yr_norm_prc", "hist_wk_norm_prc"),
        ("prev_yr_vol_wk_diff", "hist_wk_vol_diff"),
        ("prev_yr_prc_wk_diff", "hist_wk_prc_diff"),
    ]:
        daily_df[c] = daily_df[c].fillna(daily_df[fb])

    daily_df["yoy_same_period_log_vol"] = 0.50 * daily_df["prev_yr_log_vol"] + 0.50 * daily_df["hist_wk_log_vol"]
    daily_df["yoy_same_period_norm_prc"] = 0.35 * daily_df["prev_yr_norm_prc"] + 0.65 * daily_df["hist_wk_norm_prc"]
    daily_df["yoy_same_period_vol_diff"] = 0.30 * daily_df["prev_yr_vol_wk_diff"] + 0.70 * daily_df["hist_wk_vol_diff"]
    daily_df["yoy_same_period_prc_diff"] = 0.30 * daily_df["prev_yr_prc_wk_diff"] + 0.70 * daily_df["hist_wk_prc_diff"]

    daily_df["vol_diff_1w"] = (daily_df["log_vol_lag_1"] - daily_df["log_vol_lag_7"]).clip(-2.5, 2.5)
    daily_df["prc_diff_1w"] = (daily_df["norm_prc_lag_1"] - daily_df["norm_prc_lag_7"]).clip(-2.5, 2.5)
    daily_df["vol_yoy_change"] = (daily_df["log_vol_lag_1"] - daily_df["yoy_same_period_log_vol"]).clip(-2.5, 2.5)
    daily_df["prc_yoy_change"] = (daily_df["norm_prc_lag_1"] - daily_df["yoy_same_period_norm_prc"]).clip(-2.5, 2.5)
    daily_df["exp_vol_yoy_chg"] = daily_df["log_vol_lag_7"] + daily_df["yoy_same_period_vol_diff"]
    daily_df["exp_prc_yoy_chg"] = daily_df["norm_prc_lag_7"] + daily_df["yoy_same_period_prc_diff"]

    daily_df["target_vol_log"] = daily_df["log_vol"]
    daily_df["target_prc_norm"] = daily_df["norm_prc_clean"]

    train_df = daily_df[daily_df["year"] <= 2024].copy()
    val_df = daily_df[daily_df["year"] == 2025].copy()
    if len(val_df) < 100:
        split_idx = int(len(daily_df) * 0.8)
        train_df = daily_df.iloc[:split_idx].copy()
        val_df = daily_df.iloc[split_idx:].copy()

    X_tr_v, L_tr_v, mv_v, sv_v = prep_arrays(train_df, FEATURE_COLS_VOL, LAG_COLS_VOL)
    X_va_v, L_va_v, _, _ = prep_arrays(val_df, FEATURE_COLS_VOL, LAG_COLS_VOL, mv_v, sv_v)
    y_tr_v = train_df["target_vol_log"].values.astype(np.float32)
    y_va_v = val_df["target_vol_log"].values.astype(np.float32)

    X_tr_p, L_tr_p, mv_p, sv_p = prep_arrays(train_df, FEATURE_COLS_PRC, LAG_COLS_PRC)
    X_va_p, L_va_p, _, _ = prep_arrays(val_df, FEATURE_COLS_PRC, LAG_COLS_PRC, mv_p, sv_p)
    y_tr_p = train_df["target_prc_norm"].values.astype(np.float32)
    y_va_p = val_df["target_prc_norm"].values.astype(np.float32)

    XL_tr_v = np.hstack([X_tr_v, L_tr_v])
    XL_va_v = np.hstack([X_va_v, L_va_v])
    XL_tr_p = np.hstack([X_tr_p, L_tr_p])
    XL_va_p = np.hstack([X_va_p, L_va_p])

    tft_v = train_tft_model(X_tr_v, L_tr_v, y_tr_v, X_va_v, L_va_v, y_va_v, epochs=55, lr=0.0055, seed=42)
    tft_p = train_tft_model(X_tr_p, L_tr_p, y_tr_p, X_va_p, L_va_p, y_va_p, epochs=55, lr=0.0050, seed=108)

    xgb_v = train_xgb_quantiles(XL_tr_v, y_tr_v, XL_va_v, y_va_v, n_estimators=260, max_depth=6, lr=0.045, seed=42)
    xgb_p = train_xgb_quantiles(XL_tr_p, y_tr_p, XL_va_p, y_va_p, n_estimators=280, max_depth=6, lr=0.045, seed=108)

    lgb_v = train_lgb_quantiles(XL_tr_v, y_tr_v, XL_va_v, y_va_v, n_estimators=280, num_leaves=63, max_depth=9, lr=0.045, seed=42)
    lgb_p = train_lgb_quantiles(XL_tr_p, y_tr_p, XL_va_p, y_va_p, n_estimators=280, num_leaves=63, max_depth=9, lr=0.045, seed=108)

    rf_v = train_rf_quantile(XL_tr_v, y_tr_v, n_estimators=160, max_depth=18, seed=42)
    rf_p = train_rf_quantile(XL_tr_p, y_tr_p, n_estimators=160, max_depth=18, seed=108)

    X_all_v, L_all_v, _, _ = prep_arrays(daily_df, FEATURE_COLS_VOL, LAG_COLS_VOL, mv_v, sv_v)
    X_all_p, L_all_p, _, _ = prep_arrays(daily_df, FEATURE_COLS_PRC, LAG_COLS_PRC, mv_p, sv_p)
    XL_all_v = np.hstack([X_all_v, L_all_v])
    XL_all_p = np.hstack([X_all_p, L_all_p])

    def decode_vol(q_arr):
        return np.expm1(np.clip(q_arr, 0.0, 10.0))

    def decode_prc(q_arr, mean_s, std_s):
        return np.clip(q_arr * std_s[:, None] + mean_s[:, None], 300.0, None)

    p_mean = daily_df["grp_prc_mean"].values
    p_std = daily_df["grp_prc_std"].values

    hist_out = pd.DataFrame({
        "date": daily_df["date"],
        "year": daily_df["year"],
        "month": daily_df["month"],
        "week_idx": daily_df["week_idx"],
        "holiday_period": daily_df["holiday_period"],
        "group_id": daily_df["group_id"],
        "variety": daily_df["variety"],
        "box_unit": daily_df["box_unit"],
        "unit_weight_kg": daily_df["unit_weight_kg"],
        "grp_prc_mean": daily_df["grp_prc_mean"],
        "grp_prc_std": daily_df["grp_prc_std"],
        "actual_vol_ton": daily_df["volume_ton"],
        "actual_prc_kg": daily_df["unit_price_kg_clean"],
        "hist_vol_21_25": daily_df["hist_wk_vol_ton"],
        "hist_prc_21_25": daily_df["hist_wk_prc_kg"],
        "is_future": False,
    })

    for m_name, v_q, p_q in [
        ("TFT", predict_tft(tft_v, X_all_v, L_all_v), predict_tft(tft_p, X_all_p, L_all_p)),
        ("XGBoost", predict_quantile_dict(xgb_v, XL_all_v), predict_quantile_dict(xgb_p, XL_all_p)),
        ("LightGBM", predict_quantile_dict(lgb_v, XL_all_v), predict_quantile_dict(lgb_p, XL_all_p)),
        ("RandomForest", predict_rf_quantile(rf_v, XL_all_v), predict_rf_quantile(rf_p, XL_all_p)),
    ]:
        dv = decode_vol(v_q)
        dp = decode_prc(p_q, p_mean, p_std)
        hist_out[f"{m_name}_vol_p10"] = dv[:, 0]
        hist_out[f"{m_name}_vol_p50"] = dv[:, 1]
        hist_out[f"{m_name}_vol_p90"] = dv[:, 2]
        hist_out[f"{m_name}_prc_p10"] = dp[:, 0]
        hist_out[f"{m_name}_prc_p50"] = dp[:, 1]
        hist_out[f"{m_name}_prc_p90"] = dp[:, 2]

    last_dt = daily_df["date"].max()
    fut_dates = [
        d for d in pd.date_range(last_dt + pd.Timedelta(days=1), "2026-12-31", freq="D")
        if d.dayofweek != 6 and d not in korean_holidays
    ]

    fut_rows = []
    if fut_dates:
        fut_hol = compute_holiday_features(pd.Series(fut_dates)).set_index("date")
        wx_idx = daily_wx.set_index("date")
        wk_idx_map = wk_full_grid.set_index(["group_id", "week_idx"])
        prev_yr_map = prev_yr_lookup.set_index(["group_id", "year", "week_idx"])
        grp_m_map = grp_month_prof.set_index(["group_id", "month"])
        grp_h_map = grp_hol_prof.set_index(["group_id", "holiday_period"])
        grp_w_map = grp_wx_prof.set_index(["group_id", "month", "wx_rain_regime"])
        grp_p_map = grp_prof.set_index("group_id")

        for _, urow in top5_units.iterrows():
            gid = urow["group_id"]
            var = urow["variety"]
            bunit = urow["box_unit"]
            uwt = float(urow["단위중량_kg"]) if pd.notnull(urow["단위중량_kg"]) and urow["단위중량_kg"] > 0 else 5.0

            sub_g = daily_df[daily_df["group_id"] == gid].sort_values("date")
            if len(sub_g) == 0 or gid not in grp_p_map.index:
                continue

            g_vol_mean = float(grp_p_map.loc[gid, "grp_vol_mean"])
            g_prc_mean = float(grp_p_map.loc[gid, "grp_prc_mean"])
            g_prc_std = float(grp_p_map.loc[gid, "grp_prc_std"])
            log_g_vol_mean = float(np.log1p(g_vol_mean))
            norm_g_prc_mean = g_prc_mean / 5000.0
            norm_uwt = uwt / 10.0

            vol_hist = sub_g["log_vol"].tolist()[-35:]
            prc_hist = sub_g["norm_prc_clean"].tolist()[-35:]
            last_teuk = float(sub_g["ratio_teuk"].iloc[-1])

            for d in fut_dates:
                wk = min(52, (d.dayofyear - 1) // 7 + 1)
                if (var, wk) not in active_season_pairs:
                    continue

                hr = fut_hol.loc[d]
                wr = wx_idx.loc[d]
                h_per = hr["holiday_period"]
                rain_sum = float(wr["wx_rain_dur_wk1_sum"])
                rain_reg = "건조·청명(0~3h)" if rain_sum <= 3.0 else ("보통강우(3~15h)" if rain_sum <= 15.0 else "장시간강우(15h+)")

                s_vol = float(grp_m_map.loc[(gid, d.month), "seasonal_vol_mean"]) if (gid, d.month) in grp_m_map.index else g_vol_mean
                s_prc = float(grp_m_map.loc[(gid, d.month), "seasonal_prc_mean"]) if (gid, d.month) in grp_m_map.index else g_prc_mean
                h_vol = float(grp_h_map.loc[(gid, h_per), "hol_vol_mean"]) if (gid, h_per) in grp_h_map.index else s_vol
                h_prc = float(grp_h_map.loc[(gid, h_per), "hol_prc_median"]) if (gid, h_per) in grp_h_map.index else s_prc
                w_vol = float(grp_w_map.loc[(gid, d.month, rain_reg), "wx_vol_mean"]) if (gid, d.month, rain_reg) in grp_w_map.index else s_vol
                w_prc = float(grp_w_map.loc[(gid, d.month, rain_reg), "wx_prc_median"]) if (gid, d.month, rain_reg) in grp_w_map.index else s_prc

                s_log_vol = float(np.log1p(s_vol))
                norm_s_prc = (s_prc - g_prc_mean) / g_prc_std

                def get_lag(arr, k, fb):
                    return float(arr[-k]) if len(arr) >= k else fb

                lv1 = get_lag(vol_hist, 1, s_log_vol)
                lv2 = get_lag(vol_hist, 2, s_log_vol)
                lv3 = get_lag(vol_hist, 3, s_log_vol)
                lv7 = get_lag(vol_hist, 7, s_log_vol)
                lv14 = get_lag(vol_hist, 14, s_log_vol)
                lv30 = get_lag(vol_hist, 30, s_log_vol)

                np1 = get_lag(prc_hist, 1, norm_s_prc)
                np2 = get_lag(prc_hist, 2, norm_s_prc)
                np3 = get_lag(prc_hist, 3, norm_s_prc)
                np7 = get_lag(prc_hist, 7, norm_s_prc)
                np14 = get_lag(prc_hist, 14, norm_s_prc)
                np30 = get_lag(prc_hist, 30, norm_s_prc)

                wkr = wk_idx_map.loc[(gid, wk)]
                pyr = prev_yr_map.loc[(gid, d.year, wk)] if (gid, d.year, wk) in prev_yr_map.index else None
                h_wk_lv = float(wkr["hist_wk_log_vol"])
                h_wk_np = float(wkr["hist_wk_norm_prc"])
                h_wk_vd = float(wkr["hist_wk_vol_diff"])
                h_wk_pd = float(wkr["hist_wk_prc_diff"])

                py_lv = float(pyr["prev_yr_log_vol"]) if pyr is not None else h_wk_lv
                py_np = float(pyr["prev_yr_norm_prc"]) if pyr is not None else h_wk_np
                py_vd = float(pyr["prev_yr_vol_wk_diff"]) if pyr is not None else h_wk_vd
                py_pd = float(pyr["prev_yr_prc_wk_diff"]) if pyr is not None else h_wk_pd

                yoy_lv = 0.50 * py_lv + 0.50 * h_wk_lv
                yoy_np = 0.35 * py_np + 0.65 * h_wk_np
                yoy_vd = 0.30 * py_vd + 0.70 * h_wk_vd
                yoy_pd = 0.30 * py_pd + 0.70 * h_wk_pd

                feat_row = {
                    "log_grp_vol_mean": log_g_vol_mean,
                    "norm_grp_prc_mean": norm_g_prc_mean,
                    "norm_unit_weight": norm_uwt,
                    "seasonal_log_vol": s_log_vol,
                    "norm_seasonal_prc": norm_s_prc,
                    "seollal_pre_surge": float(hr["seollal_pre_surge"]),
                    "seollal_post_dip": float(hr["seollal_post_dip"]),
                    "chuseok_pre_surge": float(hr["chuseok_pre_surge"]),
                    "chuseok_post_dip": float(hr["chuseok_post_dip"]),
                    "holiday_vol_effect": float(np.clip(np.log1p(h_vol) - np.log1p(s_vol), -1.5, 1.5)),
                    "holiday_prc_effect": float(np.clip((h_prc - s_prc) / g_prc_std, -1.5, 1.5)),
                    "wx_temp_wk1_norm": float(wr["wx_temp_wk1_norm"]),
                    "wx_temp_anom_norm": float(wr["wx_temp_anom_norm"]),
                    "wx_rain_wk1_norm": float(wr["wx_rain_wk1_norm"]),
                    "wx_sun_wk1_norm": float(wr["wx_sun_wk1_norm"]),
                    "wx_snow_wk1_norm": float(wr["wx_snow_wk1_norm"]),
                    "wx_vol_effect": float(np.clip(np.log1p(w_vol) - np.log1p(s_vol), -1.5, 1.5)),
                    "wx_prc_effect": float(np.clip((w_prc - s_prc) / g_prc_std, -1.5, 1.5)),
                    "sin_doy1": math.sin(2 * math.pi * d.dayofyear / 365.25),
                    "cos_doy1": math.cos(2 * math.pi * d.dayofyear / 365.25),
                    "sin_doy2": math.sin(4 * math.pi * d.dayofyear / 365.25),
                    "cos_doy2": math.cos(4 * math.pi * d.dayofyear / 365.25),
                    "sin_dow": math.sin(2 * math.pi * d.dayofweek / 6.0),
                    "cos_dow": math.cos(2 * math.pi * d.dayofweek / 6.0),
                    "year_norm": (d.year - 2021) / 5.0,
                    "log_vol_lag_1": lv1, "log_vol_lag_2": lv2, "log_vol_lag_3": lv3,
                    "log_vol_lag_7": lv7, "log_vol_lag_14": lv14, "log_vol_lag_30": lv30,
                    "norm_prc_lag_1": np1, "norm_prc_lag_2": np2, "norm_prc_lag_3": np3,
                    "norm_prc_lag_7": np7, "norm_prc_lag_14": np14, "norm_prc_lag_30": np30,
                    "vol_diff_1w": float(np.clip(lv1 - lv7, -2.5, 2.5)),
                    "prc_diff_1w": float(np.clip(np1 - np7, -2.5, 2.5)),
                    "yoy_same_period_log_vol": yoy_lv,
                    "yoy_same_period_norm_prc": yoy_np,
                    "yoy_same_period_vol_diff": yoy_vd,
                    "yoy_same_period_prc_diff": yoy_pd,
                    "vol_yoy_change": float(np.clip(lv1 - yoy_lv, -2.5, 2.5)),
                    "prc_yoy_change": float(np.clip(np1 - yoy_np, -2.5, 2.5)),
                    "exp_vol_yoy_chg": lv7 + yoy_vd,
                    "exp_prc_yoy_chg": np7 + yoy_pd,
                    "vol_roll7_mean": float(np.mean(vol_hist[-7:])) if vol_hist else s_log_vol,
                    "vol_roll14_mean": float(np.mean(vol_hist[-14:])) if vol_hist else s_log_vol,
                    "norm_prc_roll7": float(np.mean(prc_hist[-7:])) if prc_hist else norm_s_prc,
                    "norm_prc_roll14": float(np.mean(prc_hist[-14:])) if prc_hist else norm_s_prc,
                    "ratio_teuk_lag1": last_teuk,
                }
                row_df = pd.DataFrame([feat_row])
                xv, lv_arr, _, _ = prep_arrays(row_df, FEATURE_COLS_VOL, LAG_COLS_VOL, mv_v, sv_v)
                xp, lp_arr, _, _ = prep_arrays(row_df, FEATURE_COLS_PRC, LAG_COLS_PRC, mv_p, sv_p)
                xlv = np.hstack([xv, lv_arr])
                xlp = np.hstack([xp, lp_arr])

                pm = np.array([g_prc_mean])
                ps = np.array([g_prc_std])

                rec = {
                    "date": d,
                    "year": d.year,
                    "month": d.month,
                    "week_idx": wk,
                    "holiday_period": h_per,
                    "group_id": gid,
                    "variety": var,
                    "box_unit": bunit,
                    "unit_weight_kg": uwt,
                    "grp_prc_mean": g_prc_mean,
                    "grp_prc_std": g_prc_std,
                    "actual_vol_ton": np.nan,
                    "actual_prc_kg": np.nan,
                    "hist_vol_21_25": float(wkr["hist_wk_vol_ton"]),
                    "hist_prc_21_25": float(wkr["hist_wk_prc_kg"]),
                    "is_future": True,
                }

                rf_vq_raw = predict_rf_quantile(rf_v, xlv)
                rf_pq_raw = predict_rf_quantile(rf_p, xlp)

                for m_name, v_q, p_q in [
                    ("TFT", predict_tft(tft_v, xv, lv_arr), predict_tft(tft_p, xp, lp_arr)),
                    ("XGBoost", predict_quantile_dict(xgb_v, xlv), predict_quantile_dict(xgb_p, xlp)),
                    ("LightGBM", predict_quantile_dict(lgb_v, xlv), predict_quantile_dict(lgb_p, xlp)),
                    ("RandomForest", rf_vq_raw, rf_pq_raw),
                ]:
                    dv = decode_vol(v_q)[0]
                    dp = decode_prc(p_q, pm, ps)[0]
                    rec[f"{m_name}_vol_p10"] = float(dv[0])
                    rec[f"{m_name}_vol_p50"] = float(dv[1])
                    rec[f"{m_name}_vol_p90"] = float(dv[2])
                    rec[f"{m_name}_prc_p10"] = float(dp[0])
                    rec[f"{m_name}_prc_p50"] = float(dp[1])
                    rec[f"{m_name}_prc_p90"] = float(dp[2])

                fut_rows.append(rec)
                vol_hist.append(float(0.65 * rf_vq_raw[0, 1] + 0.35 * yoy_lv))
                prc_hist.append(float(0.65 * rf_pq_raw[0, 1] + 0.35 * yoy_np))

    if fut_rows:
        all_preds_df = pd.concat([hist_out, pd.DataFrame(fut_rows)], ignore_index=True)
    else:
        all_preds_df = hist_out
    all_preds_df = all_preds_df.sort_values(["group_id", "date"]).reset_index(drop=True)

    act = all_preds_df[all_preds_df["actual_vol_ton"].notnull() & all_preds_df["actual_prc_kg"].notnull()]
    acc_strs = []
    for m_name in ["TFT", "XGBoost", "LightGBM", "RandomForest"]:
        yp = act["actual_prc_kg"].values
        pp = act[f"{m_name}_prc_p50"].values
        p_acc = float(np.clip(100.0 - np.mean(np.abs((yp - pp) / np.maximum(yp, 100.0))) * 100.0, 0, 100))
        yv = act["actual_vol_ton"].values
        pv = act[f"{m_name}_vol_p50"].values
        v_r2 = float(max(0.0, 1.0 - np.sum((yv - pv) ** 2) / (np.sum((yv - np.mean(yv)) ** 2) + 1e-8)))
        acc_strs.append(f"{m_name}(P:{p_acc:.1f}%, V_R2:{v_r2:.3f})")

    print(
        f"[{mkt_disp_name}] Done in {time.time() - t_m:.1f}s | "
        f"units={len(top5_units)}, rows={len(all_preds_df):,} (future={int(all_preds_df['is_future'].sum())}) | "
        + " | ".join(acc_strs),
        flush=True,
    )

    return {
        "all_preds_df": all_preds_df,
        "top5_units": top5_units,
        "variety_order": variety_order,
        "offseason_summary_df": offseason_summary_df,
        "active_season_pairs": active_season_pairs,
        "korean_holidays": korean_holidays,
        "wk_full_grid": wk_full_grid,
    }


def main():
    t0 = time.time()
    with open(PKL_PATH, "rb") as f:
        existing = pickle.load(f)

    if "markets" in existing and "전체시장" in existing["markets"]:
        whole_bundle = existing["markets"]["전체시장"]
    else:
        whole_bundle = {
            k: existing[k]
            for k in [
                "all_preds_df", "top5_units", "variety_order",
                "offseason_summary_df", "active_season_pairs",
                "korean_holidays", "wk_full_grid",
            ]
        }

    korean_holidays = build_korean_holidays()
    daily_wx = load_weather()
    raw_df = load_all_raw_csvs()

    markets = {"전체시장": whole_bundle}
    market_order = ["전체시장"]
    market_raw_map = {"전체시장": None}

    for disp_name, raw_name in MARKET_SPECS:
        df_m = raw_df[raw_df["whsl_mrkt_nm"] == raw_name]
        bundle = build_single_market_bundle(df_m, disp_name, daily_wx, korean_holidays)
        markets[disp_name] = bundle
        market_order.append(disp_name)
        market_raw_map[disp_name] = raw_name

    combined = dict(whole_bundle)
    combined["market_order"] = market_order
    combined["market_raw_map"] = market_raw_map
    combined["markets"] = markets

    with open(PKL_PATH, "wb") as f:
        pickle.dump(combined, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(PKL_PATH) / (1024 * 1024)
    print(f"\nSaved multi-market predictions to {PKL_PATH} ({size_mb:.2f} MB) in {time.time() - t0:.1f}s", flush=True)
    print("Included markets:", market_order, flush=True)


if __name__ == "__main__":
    main()
