import pandas as pd
import numpy as np
import holidays
import csv
import joblib
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')

print("🍊 [1/4단계] 데이터 로드 및 시장·품종·포장규격 단위 집계 시작...")

kr_holidays = holidays.KR(years=range(2015, 2030))
major_holiday_dates = sorted([
    pd.Timestamp(dt) for dt, name in kr_holidays.items()
    if any(k in name for k in ['설날', '추석', "New Year's Day", 'Seollal', 'Chuseok'])
    and not any(ex in name for ex in ['전날', '다음날', '대체', 'Holiday', 'Substitute', 'observed'])
])
major_holiday_arr = np.array(major_holiday_dates, dtype='datetime64[D]')
all_holiday_set = set(pd.to_datetime(list(kr_holidays.keys())))

def add_holiday_features(df_target, date_col):
    dates = pd.to_datetime(df_target[date_col]).values.astype('datetime64[D]')
    diff_matrix = (dates[:, None] - major_holiday_arr[None, :]).astype('timedelta64[D]').astype(int)
    d_day = diff_matrix[np.arange(len(dates)), np.argmin(np.abs(diff_matrix), axis=1)]
    df_target['명절_D_day'] = np.where((d_day >= -21) & (d_day <= 10), d_day, 99)
    df_target['명절_성수기_여부'] = ((d_day >= -14) & (d_day <= -2)).astype(int)
    df_target['명절_직후_여부'] = ((d_day >= 1) & (d_day <= 5)).astype(int)
    df_target['명절_연휴_여부'] = pd.to_datetime(df_target[date_col]).isin(all_holiday_set).astype(int)
    return df_target

FILE_PATH = 'data/kat_sale_citrus_merged.csv'
valid_rows, recovered_rows = [], []
with open(FILE_PATH, 'r', encoding='utf-8-sig', errors='replace') as f:
    reader = csv.reader(f)
    header = next(([c.strip() for c in r] for r in reader if any('trd_clcln_ymd' in str(c).lower() for c in r)), [])
    expected_cols = len(header)
    for row in reader:
        if len(row) == expected_cols: valid_rows.append(row)
        elif len(row) > expected_cols and len(row) > 1 and row[1].strip().isdigit() and len(row[1].strip()) == 8: recovered_rows.append(row[:expected_cols])
        elif len(row) > 32 and row[-32].strip().isdigit() and len(row[-32].strip()) == 8: recovered_rows.append([''] + row[-32:])

df = pd.DataFrame(valid_rows + recovered_rows, columns=header)
df.columns = df.columns.str.strip()
get_col = lambda name: next((c for c in df.columns if c.lower() == name.lower()), None)

COL_DATE, COL_VARIETY = get_col('Trd_clcln_ymd'), get_col('Gds_sclsf_nm')
COL_VOLUME, COL_TOT_AMT, COL_UNIT_QTY, COL_UNIT_NM = get_col('Unit_tot_qty'), get_col('Totprc'), get_col('Unit_qty'), get_col('Unit_nm')
COL_MARKET = get_col('Query_whsl_mrkt_nm') or get_col('Whsl_mrkt_cd')
COL_BOX = '포장단위'

data = df[[COL_MARKET, COL_VARIETY, COL_DATE, COL_VOLUME, COL_TOT_AMT, COL_UNIT_QTY, COL_UNIT_NM]].copy()
data = data[data[COL_DATE].astype(str) != COL_DATE]
data[COL_DATE] = pd.to_datetime(data[COL_DATE].astype(str), errors='coerce')
for c in [COL_VOLUME, COL_TOT_AMT, COL_UNIT_QTY]: data[c] = pd.to_numeric(data[c].astype(str).str.replace(',', ''), errors='coerce')
data = data.dropna(subset=[COL_DATE, COL_VOLUME, COL_TOT_AMT, COL_UNIT_QTY])
data = data[(data[COL_UNIT_QTY] > 0) & (data[COL_UNIT_NM].astype(str).str.strip().str.lower() == 'kg')]
data[COL_BOX] = data[COL_UNIT_QTY].apply(lambda x: f"{int(x)}" if x.is_integer() else f"{x:g}") + 'kg'

for c in [COL_MARKET, COL_VARIETY, COL_BOX]:
    data = data[~data[c].astype(str).str.contains('\ufffd', na=False)]
    data[c] = data[c].fillna('미분류').astype(str).str.strip().replace({'': '미분류', '.': '미분류', 'nan': '미분류'})

TARGET_SPECS = {'조생귤': ['5kg', '3kg', '10kg'], '한라봉': ['3kg', '9kg', '5kg'], '하우스감귤': ['3kg', '5kg', '0.5kg'], '천혜향': ['3kg', '9kg', '5kg']}
mask_target = pd.Series(False, index=data.index)
for v_n, b_l in TARGET_SPECS.items(): mask_target |= ((data[COL_VARIETY] == v_n) & (data[COL_BOX].isin(b_l)))

valid_df = data[mask_target & (data[COL_MARKET] != '미분류') & (data[COL_VOLUME] > 0) & (data[COL_TOT_AMT] > 0)].copy()

# 등급·사이즈를 제외하고 [도매시장, 품종, 포장단위] 3개 기준으로 통합 집계
group_cols = [COL_MARKET, COL_VARIETY, COL_BOX]
agg_df = valid_df.groupby(group_cols + [COL_DATE], as_index=False).agg({COL_VOLUME: 'sum', COL_TOT_AMT: 'sum'})
agg_df['kg당_평균가격'] = agg_df[COL_TOT_AMT] / agg_df[COL_VOLUME]
agg_df = agg_df.sort_values(by=group_cols + [COL_DATE]).reset_index(drop=True)
print(f" -> 통합 집계 완료! 일별 데이터: {len(agg_df):,}건")
print("🍊 [2/4단계] 최근 거래 흐름 및 작년 동기(YoY) 이력 피처 생성...")
agg_df['거래공백일수'] = agg_df.groupby(group_cols)[COL_DATE].diff().dt.days.fillna(1)
agg_df['시즌번호'] = (agg_df['거래공백일수'] >= 21).groupby([agg_df[c] for c in group_cols]).cumsum()
agg_df = agg_df[agg_df.groupby(group_cols + ['시즌번호'])[COL_DATE].transform('count') >= 3].reset_index(drop=True)

s_grp = group_cols + ['시즌번호']
agg_df['직전_거래량'] = agg_df.groupby(s_grp)[COL_VOLUME].shift(1)
agg_df['최근3일_평균거래량'] = agg_df.groupby(s_grp)[COL_VOLUME].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
agg_df['최근7일_평균거래량'] = agg_df.groupby(s_grp)[COL_VOLUME].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
agg_df['최근7일_거래량표준편차'] = agg_df.groupby(s_grp)[COL_VOLUME].transform(lambda x: x.shift(1).rolling(7, min_periods=1).std()).fillna(0)
agg_df['직전_가격'] = agg_df.groupby(s_grp)['kg당_평균가격'].shift(1)
agg_df['최근3일_평균가격'] = agg_df.groupby(s_grp)['kg당_평균가격'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
agg_df['최근7일_평균가격'] = agg_df.groupby(s_grp)['kg당_평균가격'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
agg_df = agg_df.dropna(subset=['직전_거래량', '직전_가격']).reset_index(drop=True)

grp_counts = agg_df.groupby(group_cols)[COL_DATE].transform('count')
mask_iqr = pd.Series(True, index=agg_df.index)
for col in [COL_VOLUME, 'kg당_평균가격']:
    q1, q3 = agg_df.groupby(group_cols)[col].transform(lambda s: s.quantile(0.25)), agg_df.groupby(group_cols)[col].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    mask_iqr &= (grp_counts < 15) | ((agg_df[col] >= q1 - 3.0 * iqr) & (agg_df[col] <= q3 + 3.0 * iqr))

df_norm = agg_df[mask_iqr].sort_values(by=s_grp + [COL_DATE]).reset_index(drop=True)
df_norm['출하연도'] = np.where(df_norm[COL_DATE].dt.month >= 8, df_norm[COL_DATE].dt.year, df_norm[COL_DATE].dt.year - 1)
df_norm['주차'] = df_norm[COL_DATE].dt.isocalendar().week.astype(int)

yearly_wk = df_norm.groupby(group_cols + ['출하연도', '주차'], as_index=False).agg(작년동기_평균거래량=(COL_VOLUME, 'mean'), 작년동기_평균단가=('kg당_평균가격', 'mean'))
yearly_wk['출하연도'] = yearly_wk['출하연도'] + 1
df_norm = df_norm.merge(yearly_wk, on=group_cols + ['출하연도', '주차'], how='left')
df_norm['작년동기_평균거래량'] = df_norm['작년동기_평균거래량'].fillna(df_norm['최근7일_평균거래량'])
df_norm['작년동기_평균단가'] = df_norm['작년동기_평균단가'].fillna(df_norm['최근7일_평균가격'])

yoy_stats = df_norm.groupby(group_cols + ['주차'], as_index=False).agg(역대주차_평균거래량=(COL_VOLUME, 'mean'), 역대주차_평균단가=('kg당_평균가격', 'mean'))
var_daily = df_norm.groupby([COL_MARKET, COL_VARIETY, '출하연도', COL_DATE], as_index=False)[COL_VOLUME].sum().sort_values([COL_MARKET, COL_VARIETY, '출하연도', COL_DATE])
var_daily['시장품종_7일평균'] = var_daily.groupby([COL_MARKET, COL_VARIETY, '출하연도'])[COL_VOLUME].transform(lambda x: x.rolling(7, min_periods=1).mean())
season_start = var_daily.groupby([COL_MARKET, COL_VARIETY, '출하연도'], as_index=False)[COL_DATE].min().rename(columns={COL_DATE: '실제_수확개시일'})
peak_idx = var_daily.groupby([COL_MARKET, COL_VARIETY, '출하연도'])['시장품종_7일평균'].idxmax()
season_peak = var_daily.loc[peak_idx, [COL_MARKET, COL_VARIETY, '출하연도', COL_DATE]].rename(columns={COL_DATE: '역대_피크일'})
season_peak['피크_연중일수'], season_start['개시_연중일수'] = season_peak['역대_피크일'].dt.dayofyear, season_start['실제_수확개시일'].dt.dayofyear

df_norm = df_norm.merge(season_start[[COL_MARKET, COL_VARIETY, '출하연도', '실제_수확개시일']], on=[COL_MARKET, COL_VARIETY, '출하연도'], how='left')
df_norm['실제_출하경과일'] = (df_norm[COL_DATE] - df_norm['실제_수확개시일']).dt.days.clip(lower=0)
typical_season_stats = season_start.groupby([COL_MARKET, COL_VARIETY], as_index=False)['개시_연중일수'].median().merge(season_peak.groupby([COL_MARKET, COL_VARIETY], as_index=False)['피크_연중일수'].median(), on=[COL_MARKET, COL_VARIETY], how='left')
df_norm = df_norm.merge(typical_season_stats, on=[COL_MARKET, COL_VARIETY], how='left')
df_norm['피크_거리일수'] = ((df_norm[COL_DATE].dt.dayofyear - df_norm['피크_연중일수'] + 182) % 365 - 182).fillna(0)
df_norm['피크_구간여부'] = (df_norm['피크_거리일수'].abs() <= 14).astype(int)

df_norm['월'], df_norm['요일'] = df_norm[COL_DATE].dt.month, df_norm[COL_DATE].dt.dayofweek
df_norm['연중일수_sin'], df_norm['연중일수_cos'] = np.sin(2 * np.pi * df_norm[COL_DATE].dt.dayofyear / 365.0), np.cos(2 * np.pi * df_norm[COL_DATE].dt.dayofyear / 365.0)
df_norm['시즌_경과일수'] = df_norm.groupby(s_grp).cumcount() + 1
df_norm['휴장직후_개장일'], df_norm['이월누적_가중치'] = (df_norm['거래공백일수'] >= 2).astype(int), np.log1p(df_norm['거래공백일수'])
df_norm = add_holiday_features(df_norm, COL_DATE)

df_norm = df_norm.sort_values(by=group_cols + [COL_DATE]).reset_index(drop=True)
df_norm['순번'], df_norm['전체수'] = df_norm.groupby(group_cols).cumcount(), df_norm.groupby(group_cols)[COL_DATE].transform('count')
is_train = df_norm['순번'] < (df_norm['전체수'] * 0.8)

t_stats = df_norm[is_train].groupby(group_cols + ['요일'], as_index=False).agg(요일_기준거래량=(COL_VOLUME, 'median'), 요일_기준단가=('kg당_평균가격', 'median'))
df_norm = df_norm.merge(t_stats, on=group_cols + ['요일'], how='left')
df_norm['요일_기준거래량'], df_norm['요일_기준단가'] = df_norm['요일_기준거래량'].fillna(df_norm['최근7일_평균거래량']), df_norm['요일_기준단가'].fillna(df_norm['최근7일_평균가격'])

df_norm['동적_기준거래량'] = 0.6 * df_norm['최근7일_평균거래량'] + 0.4 * df_norm['작년동기_평균거래량']
df_norm['기준거래량_log'] = np.log1p(df_norm['동적_기준거래량'])
df_norm['직전_거래량_log'], df_norm['최근3일_거래량_log'], df_norm['최근7일_거래량_log'] = np.log1p(df_norm['직전_거래량']), np.log1p(df_norm['최근3일_평균거래량']), np.log1p(df_norm['최근7일_평균거래량'])
df_norm['작년동기_거래량_log'] = np.log1p(df_norm['작년동기_평균거래량'])
df_norm['직전_대비_기준거래량비'] = (df_norm['직전_거래량'] + 1) / (df_norm['동적_기준거래량'] + 1)
df_norm['최근3일_대비_기준거래량비'] = (df_norm['최근3일_평균거래량'] + 1) / (df_norm['동적_기준거래량'] + 1)
df_norm['작년동기_대비_최근비'] = (df_norm['최근7일_평균거래량'] + 1) / (df_norm['작년동기_평균거래량'] + 1)
df_norm['거래량_변동계수'] = df_norm['최근7일_거래량표준편차'] / (df_norm['최근7일_평균거래량'] + 1)

df_norm['직전_단가_배율'], df_norm['최근3일_단가_배율'] = df_norm['직전_가격'] / (df_norm['최근7일_평균가격'] + 1), df_norm['최근3일_평균가격'] / (df_norm['최근7일_평균가격'] + 1)
df_norm['요일_단가_배율'], df_norm['작년동기_단가_배율'] = df_norm['요일_기준단가'] / (df_norm['최근7일_평균가격'] + 1), df_norm['작년동기_평균단가'] / (df_norm['최근7일_평균가격'] + 1)
df_norm['target_vol_log_diff'] = (np.log1p(df_norm[COL_VOLUME]) - df_norm['기준거래량_log']).clip(-1.5, 1.5)
df_norm['target_price_ratio'] = (df_norm['kg당_평균가격'] / (df_norm['최근7일_평균가격'] + 1)).clip(0.3, 2.5)
vol_features = [COL_MARKET, COL_VARIETY, COL_BOX, '월', '요일', '주차', '연중일수_sin', '연중일수_cos', '거래공백일수', '휴장직후_개장일', '이월누적_가중치', '시즌_경과일수', '명절_D_day', '명절_성수기_여부', '명절_직후_여부', '명절_연휴_여부', '기준거래량_log', '직전_거래량_log', '최근3일_거래량_log', '최근7일_거래량_log', '작년동기_거래량_log', '직전_대비_기준거래량비', '최근3일_대비_기준거래량비', '작년동기_대비_최근비', '거래량_변동계수', '실제_출하경과일', '피크_거리일수', '피크_구간여부']
price_features = [COL_MARKET, COL_VARIETY, COL_BOX, '월', '요일', '주차', '연중일수_sin', '연중일수_cos', '거래공백일수', '휴장직후_개장일', '시즌_경과일수', '실제_출하경과일', '피크_거리일수', '피크_구간여부', '명절_D_day', '명절_성수기_여부', '명절_직후_여부', '명절_연휴_여부', '직전_대비_기준거래량비', '거래량_변동계수', '직전_단가_배율', '최근3일_단가_배율', '요일_단가_배율', '작년동기_단가_배율']

td_grp = [COL_MARKET, COL_VARIETY]
td_df = df_norm.groupby(td_grp + [COL_DATE], as_index=False).agg(시장품종_총거래량=(COL_VOLUME, 'sum'), 실제_출하경과일=('실제_출하경과일', 'first'), 피크_거리일수=('피크_거리일수', 'first'), 피크_구간여부=('피크_구간여부', 'first')).sort_values(td_grp + [COL_DATE]).reset_index(drop=True)
td_df['공백일'] = td_df.groupby(td_grp)[COL_DATE].diff().dt.days.fillna(1)
td_df['휴장직후'] = (td_df['공백일'] >= 2).astype(int)
td_df['직전_총물량'] = td_df.groupby(td_grp)['시장품종_총거래량'].shift(1).fillna(td_df['시장품종_총거래량'])
td_df['최근7일_총물량'] = td_df.groupby(td_grp)['시장품종_총거래량'].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean()).fillna(td_df['시장품종_총거래량'])
td_df['요일'], td_df['월'] = td_df[COL_DATE].dt.dayofweek, td_df[COL_DATE].dt.month
td_df = add_holiday_features(td_df, COL_DATE)
td_cutoff = df_norm[is_train][COL_DATE].max()
td_yoil_med = td_df[td_df[COL_DATE] <= td_cutoff].groupby(td_grp + ['요일'], as_index=False)['시장품종_총거래량'].median().rename(columns={'시장품종_총거래량': '총물량_요일기준'})
td_df = td_df.merge(td_yoil_med, on=td_grp + ['요일'], how='left')
td_df['총물량_요일기준'] = td_df['총물량_요일기준'].fillna(td_df['최근7일_총물량'])
td_df['총물량_기준log'], td_df['직전_총물량log'], td_df['최근7일_총물량log'] = np.log1p(td_df['총물량_요일기준']), np.log1p(td_df['직전_총물량']), np.log1p(td_df['최근7일_총물량'])
td_df['target_td_diff'] = (np.log1p(td_df['시장품종_총거래량']) - td_df['총물량_기준log']).clip(-1.5, 1.5)
td_features = td_grp + ['월', '요일', '공백일', '휴장직후', '실제_출하경과일', '피크_거리일수', '피크_구간여부', '명절_D_day', '명절_성수기_여부', '명절_직후_여부', '총물량_기준log', '직전_총물량log', '최근7일_총물량log']

print("🌲 [3/4단계] RandomForest 모델 학습 진행...")
encoder, td_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
df_rf, td_df_rf = df_norm.copy(), td_df.copy()
df_rf[group_cols], td_df_rf[td_grp] = encoder.fit_transform(df_rf[group_cols].astype(str)), td_encoder.fit_transform(td_df_rf[td_grp].astype(str))

train_rf = df_rf[is_train].dropna(subset=vol_features + price_features + ['target_vol_log_diff', 'target_price_ratio']).copy()
test_rf = df_rf[~is_train].dropna(subset=vol_features + price_features + ['target_vol_log_diff', 'target_price_ratio']).copy()
test_rf['orig_idx'] = test_rf.index

rf_vol_bu = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=8, n_jobs=-1, verbose=1, random_state=42).fit(train_rf[vol_features], train_rf['target_vol_log_diff'])
td_train_rf = td_df_rf[td_df_rf[COL_DATE] <= td_cutoff].dropna(subset=td_features + ['target_td_diff'])
rf_vol_td = RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=8, n_jobs=-1, verbose=1, random_state=42).fit(td_train_rf[td_features], td_train_rf['target_td_diff'])
rf_price = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=8, n_jobs=-1, verbose=1, random_state=42).fit(train_rf[price_features], train_rf['target_price_ratio'])

print("\n📊 [4/4단계] 최근 날짜 20% 테스트 데이터 정확도 검증 결과")
pred_bu = np.maximum(np.expm1(test_rf['기준거래량_log'].values + rf_vol_bu.predict(test_rf[vol_features])), 0)
td_df_rf['예측_총물량'] = np.maximum(np.expm1(td_df_rf['총물량_기준log'].values + rf_vol_td.predict(td_df_rf[td_features])), 0)
test_rf = test_rf.merge(td_df_rf[td_grp + [COL_DATE, '예측_총물량', '최근7일_총물량']], on=td_grp + [COL_DATE], how='left')
safe_7d = test_rf['최근7일_총물량'].fillna(test_rf['최근7일_평균거래량']).values
pred_td = np.nan_to_num(test_rf['예측_총물량'].values * np.clip(test_rf['최근7일_평균거래량'].values / (safe_7d + 1), 0, 1), nan=pred_bu)

test_orig = df_norm.loc[test_rf['orig_idx']].reset_index(drop=True)
test_orig['예측_거래량'] = 0.6 * pred_bu + 0.4 * pred_td
test_orig['예측_단가'] = np.maximum(rf_price.predict(test_rf[price_features]) * test_rf['최근7일_평균가격'].values, 0)

for (v_name, b_name), sub in test_orig.groupby([COL_VARIETY, COL_BOX]):
    yv, pv, yp, pp = sub[COL_VOLUME].values, sub['예측_거래량'].values, sub['kg당_평균가격'].values, sub['예측_단가'].values
    print(f" - {v_name:<6} ({b_name:<4}) | 테스트 {len(sub):>5,}건 | 단가 정확도: {max(0.0, 100.0 - (np.sum(np.abs(yp - pp)) / np.sum(yp) * 100.0)):6.2f}% | 거래량 정확도: {max(0.0, 100.0 - (np.sum(np.abs(yv - pv)) / np.sum(yv) * 100.0)):6.2f}%")

recent_hist_df = df_norm.groupby(group_cols, group_keys=False).tail(30).reset_index(drop=True)
recent_td_df = td_df.groupby(td_grp, group_keys=False).tail(30).reset_index(drop=True)
test_save_cols = [COL_DATE, COL_MARKET, COL_VARIETY, COL_BOX, COL_VOLUME, '예측_거래량', 'kg당_평균가격', '예측_단가']
bundle = {'rf_vol_bu': rf_vol_bu, 'rf_vol_td': rf_vol_td, 'rf_price': rf_price, 'encoder': encoder, 'td_encoder': td_encoder, 'vol_features': vol_features, 'td_features': td_features, 'price_features': price_features, 'group_cols': group_cols, 'td_grp': td_grp, 't_stats': t_stats, 'td_yoil_med': td_yoil_med, 'yoy_stats': yoy_stats, 'typical_season_stats': typical_season_stats, 'test_results': test_orig[test_save_cols].copy(), 'recent_hist': recent_hist_df, 'recent_td': recent_td_df, 'cols': {'DATE': COL_DATE, 'MARKET': COL_MARKET, 'VARIETY': COL_VARIETY, 'BOX': COL_BOX, 'VOLUME': COL_VOLUME}}
joblib.dump(bundle, 'citrus_rf_bundle.joblib', compress=3)
print("\n✅ 완료! 등급·사이즈 통합 버전 'citrus_rf_bundle.joblib' 저장됨.")

# 각 [도매시장, 품종, 포장단위]별 학습 시작일과 종료일 저장
train_period_df = df_norm[is_train].groupby(group_cols, as_index=False)[COL_DATE].agg(학습시작일='min', 학습종료일='max')
