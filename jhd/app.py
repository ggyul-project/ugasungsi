import os
import pickle
from pathlib import Path
from datetime import date, timedelta

import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import seaborn as sns

# 주차 정보를 제외하고 출하기 시작~종료 시점을 간결하게 정리하는 함수
def format_clean_season(s):
    matches = re.findall(r'(\d{2}/\d{2})~(\d{2}/\d{2})', str(s))
    if not matches:
        return str(s)

    has_head = False
    head_end = ''
    has_tail = False
    tail_start = ''
    other_ranges = []

    for st_dt, ed_dt in matches:
        if st_dt == '01/01':
            has_head = True
            head_end = ed_dt
        elif ed_dt in ['12/30', '12/31']:
            has_tail = True
            tail_start = st_dt
        else:
            other_ranges.append(f'{st_dt} ~ {ed_dt}')

    if has_head and has_tail:
        wrap_range = f'{tail_start} ~ 익년 {head_end}'
        all_ranges = [wrap_range] + other_ranges
        return ', '.join(all_ranges)
    elif len(matches) == 1:
        return f'{matches[0][0]} ~ {matches[0][1]}'
    else:
        return ', '.join([f'{st_dt} ~ {ed_dt}' for st_dt, ed_dt in matches])

# 그래프 내부 영문 출력용 기본 폰트 설정 (OS/Docker 환경 무관 한글 깨짐 방지)
plt.rc('font', family='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110
sns.set_theme(style='whitegrid', font='DejaVu Sans')
plt.rcParams['axes.unicode_minus'] = False

MARKET_EN_MAP = {
    '전체시장': 'All Markets',
    '서울가락시장': 'Seoul Garak',
    '대구북부시장': 'Daegu Bukbu',
    '부산반여시장': 'Busan Banyeo',
    '경기구리시장': 'Gyeonggi Guri',
    '부산엄궁시장': 'Busan Eomgung',
}
VARIETY_EN_MAP = {
    '조생귤': 'Josaeng Mandarin',
    '한라봉': 'Hallabong',
    '하우스감귤': 'Greenhouse Mandarin',
    '천혜향': 'Cheonhyehyang',
    '레드향': 'Redhyang',
    '극조생감귤': 'Early Mandarin',
    '비가림감귤': 'Rain-Shelter Mandarin',
    '타이벡감귤': 'Tyvek Mandarin',
    '황금향': 'Hwanggeumhyang',
}
WEEKDAY_EN = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

def to_en_unit(unit_str: str) -> str:
    return (
        str(unit_str)
        .replace('상자', ' Box')
        .replace('파렛트', ' Pallet')
        .replace('기타', ' Other')
        .strip()
    )

st.set_page_config(
    page_title="감귤 품종·거래단위별 주간(월~토) 가격·거래량 예측 대시보드",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_CONFIG = {
    'TFT': {'label': 'GPU TFT (Temporal Fusion Transformer)', 'short': 'TFT', 'color': '#2A9D8F', 'marker': 'o'},
    'XGBoost': {'label': 'GPU XGBoost (Gradient Boosting)', 'short': 'XGBoost', 'color': '#E76F51', 'marker': 's'},
    'LightGBM': {'label': 'LightGBM (Leaf-wise GBDT)', 'short': 'LightGBM', 'color': '#457B9D', 'marker': '^'},
    'RandomForest': {'label': 'Random Forest (Quantile Forest)', 'short': 'RandomForest', 'color': '#8E44AD', 'marker': 'D'}
}
WEEKDAY_KOR = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}

def calc_accuracy_metrics(df_eval, m_name, prc_mult=1.0):
    sub = df_eval[df_eval['actual_vol_ton'].notnull() & df_eval['actual_prc_kg'].notnull()].copy()
    if len(sub) == 0:
        return None
    y_v = sub['actual_vol_ton'].values.astype(float)
    p_v = sub[f'{m_name}_vol_p50'].values.astype(float)
    y_p = sub['actual_prc_kg'].values.astype(float) * prc_mult
    p_p = sub[f'{m_name}_prc_p50'].values.astype(float) * prc_mult

    # 거래량 지표
    v_mae = float(np.mean(np.abs(y_v - p_v)))
    v_rmse = float(np.sqrt(np.mean((y_v - p_v) ** 2)))
    v_mape = float(np.mean(np.abs((y_v - p_v) / np.maximum(y_v, 0.5))) * 100.0)
    v_wmape = float(np.sum(np.abs(y_v - p_v)) / max(np.sum(y_v), 1e-6) * 100.0)
    v_acc = float(np.clip(100.0 - v_wmape, 0.0, 100.0))
    ss_res_v = float(np.sum((y_v - p_v) ** 2))
    ss_tot_v = float(np.sum((y_v - np.mean(y_v)) ** 2)) + 1e-8
    v_r2 = float(max(0.0, 1.0 - (ss_res_v / ss_tot_v))) if len(y_v) > 2 else np.nan

    # 가격 지표
    p_mae = float(np.mean(np.abs(y_p - p_p)))
    p_rmse = float(np.sqrt(np.mean((y_p - p_p) ** 2)))
    p_mape = float(np.mean(np.abs((y_p - p_p) / np.maximum(y_p, 100.0))) * 100.0)
    p_acc = float(np.clip(100.0 - p_mape, 0.0, 100.0))
    ss_res_p = float(np.sum((y_p - p_p) ** 2))
    ss_tot_p = float(np.sum((y_p - np.mean(y_p)) ** 2)) + 1e-8
    p_r2 = float(max(0.0, 1.0 - (ss_res_p / ss_tot_p))) if len(y_p) > 2 else np.nan

    return {
        'n_days': len(sub),
        'v_mae': v_mae, 'v_rmse': v_rmse, 'v_mape': v_mape, 'v_wmape': v_wmape, 'v_acc': v_acc, 'v_r2': v_r2,
        'p_mae': p_mae, 'p_rmse': p_rmse, 'p_mape': p_mape, 'p_acc': p_acc, 'p_r2': p_r2
    }

def get_pkl_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "streamlit_predictions.pkl",
        Path("streamlit_predictions.pkl"),
        Path("data") / "streamlit_predictions.pkl",
        Path("jhd") / "streamlit_predictions.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path(__file__).resolve().parent / "streamlit_predictions.pkl"

@st.cache_data(show_spinner="전체시장 및 5대 도매시장 4개 모델 예측 데이터셋(2021-01-02 ~ 2026-12-31)을 불러오는 중입니다...")
def load_prediction_data(file_mtime: float):
    pkl_path = get_pkl_path()
    if not pkl_path.exists():
        raise FileNotFoundError(f"예측 데이터 파일이 존재하지 않습니다: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if "markets" in data and isinstance(data["markets"], dict):
        market_order = data.get("market_order", list(data["markets"].keys()))
        raw_markets = data["markets"]
    else:
        market_order = ["전체시장"]
        raw_markets = {"전체시장": data}

    markets_prepared = {}
    for mkt_name in market_order:
        if mkt_name not in raw_markets:
            continue
        m_data = raw_markets[mkt_name]
        df = m_data['all_preds_df'].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= '2021-01-02') & (df['date'] <= '2026-12-31')].sort_values(['group_id', 'date']).reset_index(drop=True)

        overall_acc = {}
        for m_name in MODEL_CONFIG:
            overall_acc[m_name] = calc_accuracy_metrics(df, m_name, prc_mult=1.0)

        markets_prepared[mkt_name] = {
            'all_preds_df': df,
            'top5_units': m_data['top5_units'],
            'variety_order': m_data['variety_order'],
            'offseason_summary_df': m_data['offseason_summary_df'],
            'active_season_pairs': m_data['active_season_pairs'],
            'korean_holidays': set(pd.to_datetime(list(m_data['korean_holidays']))),
            'wk_full_grid': m_data['wk_full_grid'],
            'overall_acc': overall_acc,
        }

    return market_order, markets_prepared

try:
    _pkl_path = get_pkl_path()
    _pkl_mtime = _pkl_path.stat().st_mtime if _pkl_path.exists() else 0.0
    market_order, markets_prepared = load_prediction_data(_pkl_mtime)
except Exception as e:
    st.error(f"데이터 로드 오류: {e}")
    st.stop()

# 상단 타이틀 헤더
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1D3557 0%, #2A9D8F 100%); padding: 18px 24px; border-radius: 12px; color: white; margin-bottom: 16px;">
        <h2 style="margin: 0; color: white; font-weight: 800;">🍊 감귤 4대 핵심 품종(조생귤 · 한라봉 · 하우스감귤 · 천혜향) 시장별 주간(월~토) 예상 가격 · 거래량</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# 사이드바 컨트롤
with st.sidebar:
    st.header("🔍 예측 조회 조건 설정")

    selected_market = st.selectbox(
        "1️⃣ 도매시장 선택",
        options=market_order,
        index=0
    )

    mkt_bundle = markets_prepared[selected_market]
    all_preds_df = mkt_bundle['all_preds_df']
    top5_units = mkt_bundle['top5_units']
    variety_order = mkt_bundle['variety_order']
    offseason_summary_df = mkt_bundle['offseason_summary_df']
    active_season_pairs = mkt_bundle['active_season_pairs']
    korean_holidays = mkt_bundle['korean_holidays']
    wk_full_grid = mkt_bundle['wk_full_grid']
    overall_acc_dict = mkt_bundle['overall_acc']
    n_total_groups = len(top5_units)

    TARGET_VARIETIES = ['조생귤', '한라봉', '하우스감귤', '천혜향']
    avail_varieties = [v for v in TARGET_VARIETIES if v in top5_units['variety'].unique()]
    selected_variety = st.selectbox(
        "2️⃣ 품종 선택",
        options=avail_varieties,
        index=0
    )

    sub_units = top5_units[top5_units['variety'] == selected_variety].sort_values('품종내_순위')
    unit_options = sub_units['box_unit'].tolist()
    unit_labels = {
        r['box_unit']: f"{r['box_unit']} (품종내 {r['품종내_건수비중(%)']:.1f}% · {int(r['품종내_순위'])}위)"
        for _, r in sub_units.iterrows()
    }
    selected_unit = st.selectbox(
        "3️⃣ 거래단위 선택",
        options=unit_options,
        format_func=lambda u: unit_labels.get(u, u),
        index=0
    )

    selected_gid = f"{selected_variety}_{selected_unit}"
    gid_df = all_preds_df[all_preds_df['group_id'] == selected_gid].sort_values('date').reset_index(drop=True)
    unit_weight_kg = float(gid_df['unit_weight_kg'].iloc[0]) if len(gid_df) > 0 else 5.0

    off_row = offseason_summary_df[offseason_summary_df['품종명'] == selected_variety].iloc[0]
    season_clean_str = format_clean_season(off_row['출하기(예측 수행 기간)'])
    actual_sub = gid_df[gid_df['actual_vol_ton'].notnull()]
    last_actual_dt = actual_sub['date'].max().date() if len(actual_sub) > 0 else date(2026, 11, 11)

    st.info(
        f"**[{selected_market} · {selected_variety}] 시즌 안내**\n"
        f"- **출하기 (시작~종료)**: `{season_clean_str}`"
    )

    min_dt = date(2021, 1, 2)
    max_dt = date(2026, 12, 31)

    avail_dates_list = gid_df['date'].dt.date.tolist()
    default_target = last_actual_dt if last_actual_dt in avail_dates_list else (avail_dates_list[-1] if avail_dates_list else date(2026, 11, 11))

    selected_date = st.date_input(
        "4️⃣ 기준 날짜 선택 (해당 주간 월~토 예측)",
        value=default_target,
        min_value=min_dt,
        max_value=max_dt
    )

    window_mode = "선택일 포함 주간 (월요일~토요일 6일간)"
    price_unit_mode = "kg당 단가 (원/kg)"
    selected_models = list(MODEL_CONFIG.keys())

    # 실제가격, 예년(2021~2025) 데이터, 불확실성 예측 구간(P10~P90) 항상 기본 표시
    show_actual = True
    show_hist_21_25 = True
    show_quantile_band = True

# 선택된 날짜가 포함된 주의 월요일부터 토요일까지 6일간 데이터 구성 함수
def build_mon_to_sat_window_df(gid_df, selected_variety, selected_gid, target_dt):
    target_ts = pd.Timestamp(target_dt)
    gid_indexed = gid_df.set_index('date')
    wk_hist_sub = wk_full_grid[wk_full_grid['group_id'] == selected_gid].set_index('week_idx')

    # 해당 주간의 월요일 계산 (월:0, 화:1, ..., 일:6)
    mon_ts = target_ts - pd.Timedelta(days=target_ts.dayofweek)
    # 월요일부터 토요일까지 6일간 날짜 생성 (일요일은 도매시장 정기 휴장이므로 제외)
    window_dates = [mon_ts + pd.Timedelta(days=i) for i in range(6)]

    rows = []
    for d_ts in window_dates:
        dow = d_ts.dayofweek
        wk_num = min(52, (d_ts.dayofyear - 1) // 7 + 1)
        is_sun_or_hol = (dow == 6) or (d_ts in korean_holidays)
        is_active_wk = (selected_variety, wk_num) in active_season_pairs

        h_vol = float(wk_hist_sub.loc[wk_num, 'hist_wk_vol_ton']) if wk_num in wk_hist_sub.index else np.nan
        h_prc = float(wk_hist_sub.loc[wk_num, 'hist_wk_prc_kg']) if wk_num in wk_hist_sub.index else np.nan

        if d_ts in gid_indexed.index:
            r = gid_indexed.loc[d_ts]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            row_dict = r.to_dict()
            row_dict['date'] = d_ts
            has_act = pd.notnull(r.get('actual_vol_ton', np.nan))
            row_dict['day_status'] = '정상 영업일(실제+예측)' if has_act else '정상 영업일(예측)'
        else:
            if is_sun_or_hol:
                status_str = '휴장일(공휴일)'
            elif not is_active_wk:
                status_str = f'계절적 비수기({wk_num}주차 예측제외)'
            else:
                status_str = '개장일(인접 영업일 보간)'

            row_dict = {
                'date': d_ts,
                'year': d_ts.year,
                'month': d_ts.month,
                'week_idx': wk_num,
                'holiday_period': '휴장/비수기' if (is_sun_or_hol or not is_active_wk) else '평시',
                'group_id': selected_gid,
                'variety': selected_variety,
                'box_unit': selected_unit,
                'unit_weight_kg': unit_weight_kg,
                'hist_vol_21_25': 0.0 if is_sun_or_hol else h_vol,
                'hist_prc_21_25': np.nan if is_sun_or_hol else h_prc,
                'actual_vol_ton': np.nan,
                'actual_prc_kg': np.nan,
                'is_future': d_ts > pd.Timestamp('2026-11-11'),
                'day_status': status_str
            }
            if not is_sun_or_hol and is_active_wk and len(gid_df) > 0:
                nearest_idx = (gid_df['date'] - d_ts).abs().idxmin()
                nr = gid_df.loc[nearest_idx]
                for m_name in MODEL_CONFIG:
                    for suf in ['vol_p10', 'vol_p50', 'vol_p90', 'prc_p10', 'prc_p50', 'prc_p90']:
                        row_dict[f'{m_name}_{suf}'] = float(nr[f'{m_name}_{suf}'])
            else:
                for m_name in MODEL_CONFIG:
                    for suf in ['vol_p10', 'vol_p50', 'vol_p90']:
                        row_dict[f'{m_name}_{suf}'] = 0.0 if is_sun_or_hol else np.nan
                    for suf in ['prc_p10', 'prc_p50', 'prc_p90']:
                        row_dict[f'{m_name}_{suf}'] = np.nan

        row_dict['요일'] = WEEKDAY_KOR[dow]
        row_dict['date_label'] = f"{d_ts.strftime('%m/%d')}({WEEKDAY_KOR[dow]})"
        row_dict['is_center'] = (d_ts == target_ts)
        rows.append(row_dict)

    w_df = pd.DataFrame(rows)
    if not w_df['is_center'].any() and len(w_df) > 0:
        if target_ts.dayofweek == 6:
            w_df.loc[len(w_df) - 1, 'is_center'] = True  # 일요일 선택 시 토요일 강조
        else:
            w_df.loc[0, 'is_center'] = True
    return w_df

win_df = build_mon_to_sat_window_df(gid_df, selected_variety, selected_gid, selected_date)
center_idx = int(win_df.index[win_df['is_center']][0])
center_row = win_df.loc[center_idx]
target_dow_str = WEEKDAY_KOR[pd.Timestamp(selected_date).dayofweek]
center_date_str = pd.Timestamp(selected_date).strftime('%Y-%m-%d') + f" ({target_dow_str})"
mon_str = win_df.iloc[0]['date'].strftime('%m/%d')
sat_str = win_df.iloc[-1]['date'].strftime('%m/%d')
week_range_str = f"{mon_str}(월) ~ {sat_str}(토)"

prc_mult = 1.0  # kg당 단가 고정
prc_unit_str = "원/kg"
has_any_actual = show_actual and (win_df['actual_vol_ton'].notnull().any() or win_df['actual_prc_kg'].notnull().any())

# 선택 품종·단위 및 현재 주간(월~토) 구간의 모델별 정확도 계산
group_acc_dict = {}
window_acc_dict = {}
for m_name in MODEL_CONFIG:
    group_acc_dict[m_name] = calc_accuracy_metrics(gid_df, m_name, prc_mult=prc_mult)
    window_acc_dict[m_name] = calc_accuracy_metrics(win_df, m_name, prc_mult=prc_mult)

# 모델별 정확도 분석 데이터프레임 사전 계산
acc_rows = []
for m_name in selected_models:
    m_cfg = MODEL_CONFIG[m_name]
    g_acc = group_acc_dict.get(m_name)
    w_acc = window_acc_dict.get(m_name)
    o_acc = overall_acc_dict.get(m_name)

    if g_acc and o_acc:
        acc_rows.append({
            '모델명': m_cfg['short'],
            '모델 상세': m_cfg['label'],
            f'[{selected_variety}-{selected_unit}] 가격 정확도(%)': f"{g_acc['p_acc']:.2f}%",
            f'[{selected_variety}-{selected_unit}] 가격 MAE({prc_unit_str})': f"{g_acc['p_mae']:,.0f} 원",
            f'[{selected_variety}-{selected_unit}] 가격 MAPE(%)': f"{g_acc['p_mape']:.2f}%",
            f'[{selected_variety}-{selected_unit}] 거래량 정확도(%)': f"{g_acc['v_acc']:.2f}%",
            f'[{selected_variety}-{selected_unit}] 거래량 R²': f"{g_acc['v_r2']:.4f}" if pd.notnull(g_acc['v_r2']) else "-",
            f'[{selected_variety}-{selected_unit}] 거래량 MAE(톤/일)': f"{g_acc['v_mae']:,.2f} 톤",
            '해당 주간(월~토) 가격 정확도(%)': f"{w_acc['p_acc']:.2f}% (MAE {w_acc['p_mae']:,.0f}원)" if (w_acc and w_acc['n_days'] >= 1) else "미래구간(실제없음)",
            '해당 주간(월~토) 거래량 정확도(%)': f"{w_acc['v_acc']:.2f}% (MAE {w_acc['v_mae']:,.2f}톤)" if (w_acc and w_acc['n_days'] >= 1) else "미래구간(실제없음)",
            f'전체({n_total_groups}개 규격) 가격 정확도(%)': f"{o_acc['p_acc']:.2f}% (R²={o_acc['p_r2']:.4f})",
            f'전체({n_total_groups}개 규격) 거래량 R² / 정확도': f"R²={o_acc['v_r2']:.4f} ({o_acc['v_acc']:.2f}%)",
            '_g_p_acc': g_acc['p_acc'],
            '_g_v_acc': g_acc['v_acc'],
            '_o_p_acc': o_acc['p_acc'],
            '_o_v_r2_pct': o_acc['v_r2'] * 100.0,
            '_w_p_acc': w_acc['p_acc'] if (w_acc and w_acc['n_days'] >= 1) else np.nan,
            '_w_v_acc': w_acc['v_acc'] if (w_acc and w_acc['n_days'] >= 1) else np.nan,
            '_color': m_cfg['color']
        })

acc_df = pd.DataFrame(acc_rows)

# 해당 주간(월~토) 상세 데이터 비교 테이블 사전 계산
table_rows = []
for _, r in win_df.iterrows():
    row_item = {
        '기준구분': '★ 선택 기준일' if r['is_center'] else '-',
        '날짜': r['date'].strftime('%Y-%m-%d'),
        '요일': r['요일'],
        '영업상태': r['day_status'],
        '명절구간': r['holiday_period'],
        '실제 거래량': f"{r['actual_vol_ton']:,.1f} 톤" if pd.notnull(r.get('actual_vol_ton')) else "-",
        f'실제 가격({prc_unit_str})': f"{r['actual_prc_kg']*prc_mult:,.0f} 원" if pd.notnull(r.get('actual_prc_kg')) else "-",
        '예년(21~25) 물량': f"{r['hist_vol_21_25']:,.1f} 톤" if pd.notnull(r['hist_vol_21_25']) and r['hist_vol_21_25'] > 0 else "-",
        f'예년(21~25) 가격({prc_unit_str})': f"{r['hist_prc_21_25']*prc_mult:,.0f} 원" if pd.notnull(r['hist_prc_21_25']) else "-"
    }

    for m_name in selected_models:
        m_short = MODEL_CONFIG[m_name]['short']
        vp50 = r.get(f'{m_name}_vol_p50', np.nan)
        pp50 = r.get(f'{m_name}_prc_p50', np.nan)
        pp10 = r.get(f'{m_name}_prc_p10', np.nan)
        pp90 = r.get(f'{m_name}_prc_p90', np.nan)
        row_item[f'{m_short} 예상물량'] = f"{vp50:,.1f} 톤" if pd.notnull(vp50) and vp50 > 0 else "휴장/제외"
        row_item[f'{m_short} 예상가격({prc_unit_str})'] = f"{pp50*prc_mult:,.0f} 원 ({pp10*prc_mult:,.0f}~{pp90*prc_mult:,.0f})" if pd.notnull(pp50) else "휴장/제외"

    table_rows.append(row_item)

table_df = pd.DataFrame(table_rows)
csv_bytes = table_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

# 계절적 비수기 판별 (선택 날짜 및 주간 기준)
selected_ts = pd.Timestamp(selected_date)
selected_wk = min(52, (selected_ts.dayofyear - 1) // 7 + 1)
is_selected_offseason = (selected_variety, selected_wk) not in active_season_pairs
is_center_offseason = ("비수기" in str(center_row.get('day_status', ''))) or \
                      ((selected_variety, int(center_row.get('week_idx', selected_wk))) not in active_season_pairs)
is_offseason = is_selected_offseason or is_center_offseason

# 비수기인 경우: 예측 Graph 숨김 및 대형 비수기 안내 배너 표시
if is_offseason:
    selected_date_disp = selected_date.strftime('%Y-%m-%d') + f" ({WEEKDAY_KOR[selected_date.weekday()]})"
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 50%, #FFE082 100%);
                    border: 3.5px solid #FF8F00; border-radius: 16px; padding: 38px 28px;
                    text-align: center; margin: 15px 0 30px 0; box-shadow: 0 6px 20px rgba(255, 143, 0, 0.18);">
            <div style="font-size: 64px; line-height: 1; margin-bottom: 14px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.15));">
                🚫 ❄️ 🍊
            </div>
            <div style="display: inline-block; background-color: #D84315; color: #FFFFFF; font-size: 14px; font-weight: 800; padding: 4px 16px; border-radius: 20px; letter-spacing: 1px; margin-bottom: 12px; text-transform: uppercase;">
                OFF-SEASON NOTICE
            </div>
            <h1 style="color: #BF360C; font-size: 36px; font-weight: 900; margin: 0 0 16px 0; letter-spacing: -1px;">
                ⚠️ 현재 선택일은 [{selected_market} · {selected_variety}] 계절적 비수기입니다
            </h1>
            <p style="color: #4E342E; font-size: 19px; font-weight: 700; margin: 0 0 22px 0; line-height: 1.5;">
                선택하신 날짜 <span style="background-color: #FFFFFF; padding: 4px 14px; border-radius: 8px; border: 2px solid #FF8F00; color: #C62828; font-weight: 800;">{selected_date_disp}</span>는 
                <span style="color: #1B5E20; text-decoration: underline;">[{selected_market} · {selected_variety}]</span> 품종의 
                <strong style="color: #D84315;">경매 비수기(예측 제외 기간)</strong>에 해당하여 
                <span style="background-color: #FFCDD2; color: #B71C1C; padding: 2px 8px; border-radius: 4px; font-weight: 800;">예측 Graph 출력을 비활성화(숨김)</span>합니다.
            </p>
            <div style="background-color: #FFFFFF; border-radius: 12px; padding: 22px 28px; display: inline-block; text-align: left; max-width: 780px; border: 2px solid #FFCA28; box-shadow: 0 3px 10px rgba(0,0,0,0.05);">
                <div style="font-size: 16px; color: #212529; line-height: 2.0;">
                    <div>🌿 <strong style="color: #1B5E20;">정상 출하기 (시작~종료):</strong> <span style="background-color: #E8F5E9; color: #1B5E20; font-weight: 800; font-size: 17px; padding: 2px 10px; border-radius: 6px; border: 1px solid #A5D6A7;">{season_clean_str}</span></div>
                    <div>❄️ <strong style="color: #C62828;">계절적 비수기:</strong> <span style="background-color: #FFEBEE; color: #C62828; font-weight: 800; font-size: 16px; padding: 2px 10px; border-radius: 6px; border: 1px solid #FFCDD2;">위 출하기 외 기간</span></div>
                </div>
                <hr style="margin: 14px 0; border: none; border-top: 1.5px dashed #FFB300;" />
                <div style="font-size: 14px; color: #5D4037; line-height: 1.75;">
                    💡 <strong>비수기 예측 제외 사유:</strong><br/>
                    감귤 도매시장 경매 특성상 비수기에는 출하 및 거래량이 거의 없어(0톤에 수렴) 가격 변동성이 왜곡되며 모델 신뢰도를 보장하기 어렵습니다. 이에 따라 <strong>비수기에는 인공지능 가격 및 거래량 예측 그래프(1번 및 2번) 출력을 비활성화</strong>합니다.<br/><br/>
                    👉 <strong>예측 그래프 확인 방법:</strong><br/>
                    좌측 사이드바의 <strong>[4️⃣ 기준 날짜 선택]</strong>에서 위 <strong>출하기 기간 ({season_clean_str})</strong> 내 날짜를 선택하시면 4개 모델(TFT, XGBoost, LightGBM, Random Forest)의 해당 주간(월~토) 예상 가격 및 거래량 그래프를 즉시 확인하실 수 있습니다.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("💡 비수기 기간 중에는 예측 그래프가 숨김 처리됩니다. 아래 접이식 메뉴를 클릭하시면 해당 품종의 AI 모델 전체 정확도 평가 지표와 해당 주간(월~토) 상세 상태표를 확인하실 수 있습니다.")

    with st.expander(f"📊 참고: [{selected_variety}] 품종 AI 모델별 전체 정확도(Accuracy · R² · MAE · MAPE) 평가 지표 확인", expanded=False):
        disp_acc_cols = [c for c in acc_df.columns if not c.startswith('_')]
        st.dataframe(acc_df[disp_acc_cols], use_container_width=True, hide_index=True)

    with st.expander(f"📋 참고: [{selected_variety} - {selected_unit}] 선택일 포함 주간({week_range_str}) 일정 및 비수기 상태표 확인", expanded=False):
        st.dataframe(table_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 해당 주간(월~토) 상태표 CSV 다운로드",
            data=csv_bytes,
            file_name=f"citrus_week_offseason_{selected_gid}_{selected_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="btn_download_offseason"
        )

else:
    # 정상 출하기인 경우: 휴장일 알림, 상단 KPI 요약 카드, 예측 그래프 1&2, 정확도 그래프 3, 상세표 4 전체 정상 출력
    if "휴장일" in str(center_row.get('day_status', '')):
        st.info(
            f"ℹ️ 선택한 기준일 **{center_date_str}**은 **도매시장 경매 휴장일(공휴일/일요일)**입니다. "
            f"해당 주간({week_range_str})의 개장 영업일(월~토) 실제·예상 가격 및 거래량을 함께 출력합니다."
        )

    # 1. 상단 KPI 요약 카드 (실제값 + 예년값 + 가격 정확도 1위 Random Forest 예상값 요약)
    st.subheader(f"📌 [{selected_market} · {selected_variety} - {selected_unit}] {week_range_str} 주간 요약 (선택일: {center_date_str}, 최고 정확도 모델: Random Forest)")

    n_extra_kpis = (1 if has_any_actual else 0) + 1
    kpi_cols = st.columns(1 + n_extra_kpis)
    col_cursor = 0

    h_vol_c = center_row['hist_vol_21_25'] if pd.notnull(center_row['hist_vol_21_25']) and center_row['hist_vol_21_25'] > 0 else win_df['hist_vol_21_25'].replace(0, np.nan).mean()
    h_prc_c = (center_row['hist_prc_21_25'] if pd.notnull(center_row['hist_prc_21_25']) else win_df['hist_prc_21_25'].mean()) * prc_mult

    act_vol_c = center_row['actual_vol_ton'] if pd.notnull(center_row.get('actual_vol_ton')) else win_df['actual_vol_ton'].mean()
    act_prc_c = (center_row['actual_prc_kg'] if pd.notnull(center_row.get('actual_prc_kg')) else win_df['actual_prc_kg'].mean()) * prc_mult

    if has_any_actual:
        with kpi_cols[col_cursor]:
            act_lbl = "기준일 실제" if pd.notnull(center_row.get('actual_prc_kg')) else "주간 실제평균"
            st.metric(
                label=f"🎯 실제 경매 결과 ({act_lbl})",
                value=f"{act_prc_c:,.0f} {prc_unit_str}" if pd.notnull(act_prc_c) else "-",
                delta=f"실제 거래량: {act_vol_c:,.1f} 톤/일" if pd.notnull(act_vol_c) else "-",
                delta_color="off"
            )
            act_days_cnt = int(win_df['actual_vol_ton'].notnull().sum())
            st.caption(f"해당 주간 실제 경매일: {act_days_cnt}일")
        col_cursor += 1

    with kpi_cols[col_cursor]:
        st.metric(
            label="📊 예년(2021~2025) 동시점 평균",
            value=f"{h_prc_c:,.0f} {prc_unit_str}" if pd.notnull(h_prc_c) else "-",
            delta=f"예년 물량: {h_vol_c:,.1f} 톤/일" if pd.notnull(h_vol_c) else "-",
            delta_color="off"
        )
        st.caption(f"해당 시점({int(center_row['week_idx'])}주차) 5개년 기준")
    col_cursor += 1

    best_m = 'RandomForest'
    best_cfg = MODEL_CONFIG[best_m]
    with kpi_cols[col_cursor]:
        c_vol = center_row.get(f'{best_m}_vol_p50', np.nan)
        c_prc = center_row.get(f'{best_m}_prc_p50', np.nan)
        label_suffix = "기준일"
        if pd.isnull(c_prc) or c_vol == 0.0:
            c_vol = win_df[f'{best_m}_vol_p50'].replace(0, np.nan).mean()
            c_prc = win_df[f'{best_m}_prc_p50'].mean()
            label_suffix = "주간 평균"
        c_prc_disp = c_prc * prc_mult if pd.notnull(c_prc) else np.nan
        if has_any_actual and pd.notnull(act_prc_c) and act_prc_c > 0 and pd.notnull(c_prc_disp):
            diff_str = f"{(c_prc_disp / act_prc_c - 1.0)*100.0:+.1f}% vs 실제"
        elif pd.notnull(c_prc_disp) and pd.notnull(h_prc_c) and h_prc_c > 0:
            diff_str = f"{(c_prc_disp / h_prc_c - 1.0)*100.0:+.1f}% vs 예년"
        else:
            diff_str = "-"
        st.metric(
            label=f"🌲 {best_cfg['short']} 예상 ({label_suffix})",
            value=f"{c_prc_disp:,.0f} {prc_unit_str}" if pd.notnull(c_prc_disp) else "-",
            delta=f"예상 물량: {c_vol:,.1f} 톤 ({diff_str})" if pd.notnull(c_vol) else "-"
        )
        rf_overall_p_acc = overall_acc_dict['RandomForest']['p_acc'] if overall_acc_dict.get('RandomForest') else 0.0
        st.caption(f"가격 정확도 1위 모델 ({selected_market} 2021~2026 전체 {rf_overall_p_acc:.1f}%)")
    col_cursor += 1

    st.divider()

    # 2. 메인 비교 그래프: 선택일 포함 주간(월~토) 실제 vs 예년(21~25) vs Random Forest 예상 거래량 · 가격 비교 그래프
    st.subheader(f"📈 1. 선택일 포함 주간({week_range_str}) 실제 vs 예년(21~25) vs Random Forest 예상 거래량 · 가격 비교 그래프")

    mkt_en = MARKET_EN_MAP.get(selected_market, selected_market)
    var_en = VARIETY_EN_MAP.get(selected_variety, selected_variety)
    unit_en = to_en_unit(selected_unit)
    week_range_en = f"{mon_str}(Mon) ~ {sat_str}(Sat)"
    prc_unit_en = "KRW/kg"

    fig_comp, (ax_vol, ax_prc) = plt.subplots(1, 2, figsize=(18.5, 6.2))
    x_idx = np.arange(len(win_df))
    x_labels = [
        f"* {r['date'].strftime('%m/%d')}({WEEKDAY_EN[r['date'].dayofweek]})\n[Selected]"
        if r['is_center']
        else (
            f"{r['date'].strftime('%m/%d')}({WEEKDAY_EN[r['date'].dayofweek]})\n(Closed)"
            if "휴장" in str(r['day_status'])
            else f"{r['date'].strftime('%m/%d')}({WEEKDAY_EN[r['date'].dayofweek]})"
        )
        for _, r in win_df.iterrows()
    ]

    for ax in (ax_vol, ax_prc):
        ax.axvspan(center_idx - 0.42, center_idx + 0.42, color='#FFD166', alpha=0.30, zorder=0, label='Selected Date')
        ax.axvline(center_idx, color='#D62828', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)

    # (좌) 일주일간 거래량 비교 그래프 (실제 vs 예년 vs Random Forest 단일 모델만 출력)
    n_bars = 1 + (1 if show_hist_21_25 else 0) + (1 if has_any_actual else 0)
    bar_w = 0.22 if n_bars == 3 else (0.30 if n_bars == 2 else 0.45)
    offsets = np.linspace(-bar_w * (n_bars - 1) / 2, bar_w * (n_bars - 1) / 2, n_bars)
    b_pos = 0

    if show_hist_21_25:
        ax_vol.bar(
            x_idx + offsets[b_pos],
            win_df['hist_vol_21_25'].fillna(0.0),
            width=bar_w, color='#8ECAE6', edgecolor='#219EBC', alpha=0.85,
            label='2021-2025 Avg Volume (ton)'
        )
        b_pos += 1

    if has_any_actual:
        act_v_arr = win_df['actual_vol_ton'].fillna(0.0).values
        ax_vol.bar(
            x_idx + offsets[b_pos],
            act_v_arr,
            width=bar_w, color='#1D3557', edgecolor='black', linewidth=0.8, alpha=0.92,
            label='Actual Volume (ton)'
        )
        ax_vol.plot(
            x_idx + offsets[b_pos],
            win_df['actual_vol_ton'].values,
            color='#1D3557', marker='*', linewidth=2.0, markersize=8, zorder=5
        )
        if act_v_arr[center_idx] > 0:
            ax_vol.text(
                center_idx + offsets[b_pos],
                act_v_arr[center_idx] * 1.04,
                f"Act {act_v_arr[center_idx]:,.1f}t",
                ha='center', va='bottom', fontsize=8.2, fontweight='bold', color='#1D3557'
            )
        b_pos += 1

    rf_v_vals = win_df[f'{best_m}_vol_p50'].fillna(0.0).values
    ax_vol.bar(
        x_idx + offsets[b_pos],
        rf_v_vals,
        width=bar_w, color=best_cfg['color'], alpha=0.90,
        label=f"{best_cfg['short']} Forecast Volume"
    )
    ax_vol.plot(
        x_idx + offsets[b_pos],
        np.where(rf_v_vals > 0, rf_v_vals, np.nan),
        color=best_cfg['color'], marker=best_cfg['marker'], linewidth=1.8, markersize=6
    )
    if rf_v_vals[center_idx] > 0:
        ax_vol.text(
            center_idx + offsets[b_pos],
            rf_v_vals[center_idx] * 1.02,
            f"{rf_v_vals[center_idx]:,.1f}t",
            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=best_cfg['color']
        )

    ax_vol.set_title(f"[{mkt_en} | {var_en} - {unit_en}] Weekly ({week_range_en}) Volume: Actual vs 2021-2025 vs {best_cfg['short']} (ton/day)", fontsize=11.0, fontweight='bold', pad=10)
    ax_vol.set_xticks(x_idx)
    ax_vol.set_xticklabels(x_labels, fontsize=9.5, fontweight='bold')
    ax_vol.set_ylabel("Volume (ton/day)", fontsize=10, fontweight='bold')
    ax_vol.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.1f}'))
    max_vol_candidates = [win_df[f'{best_m}_vol_p90'].max(), win_df['hist_vol_21_25'].max(), win_df['actual_vol_ton'].max(), 1.0]
    max_vol_y = np.nanmax([v for v in max_vol_candidates if pd.notnull(v)])
    if np.isfinite(max_vol_y) and max_vol_y > 0:
        ax_vol.set_ylim(0, max_vol_y * 1.28)
    ax_vol.legend(loc='upper right', fontsize=8.5)

    # (우) 일주일간 가격 비교 그래프 (실제 vs 예년 vs Random Forest 단일 모델만 출력)
    if show_hist_21_25:
        ax_prc.plot(
            x_idx,
            win_df['hist_prc_21_25'] * prc_mult,
            color='#6C757D', linestyle=':', linewidth=2.0, marker='D', markersize=5.8,
            label=f"2021-2025 Avg Price ({prc_unit_en})"
        )

    if has_any_actual:
        act_p_vals = win_df['actual_prc_kg'] * prc_mult
        ax_prc.plot(
            x_idx,
            act_p_vals,
            color='#1D3557', linestyle='-', linewidth=2.8, marker='*', markersize=9.0, zorder=6,
            label=f"Actual Price ({prc_unit_en})"
        )
        if pd.notnull(act_p_vals.iloc[center_idx]):
            ax_prc.annotate(
                f"Actual: {act_p_vals.iloc[center_idx]:,.0f} KRW",
                xy=(center_idx, act_p_vals.iloc[center_idx]),
                xytext=(0, -15), textcoords="offset points",
                ha='center', fontsize=8.8, fontweight='bold', color='#1D3557'
            )

    rf_p50 = win_df[f'{best_m}_prc_p50'] * prc_mult
    rf_p10 = win_df[f'{best_m}_prc_p10'] * prc_mult
    rf_p90 = win_df[f'{best_m}_prc_p90'] * prc_mult
    ax_prc.plot(
        x_idx, rf_p50,
        color=best_cfg['color'], linestyle='--', linewidth=2.4, marker=best_cfg['marker'], markersize=7.0,
        label=f"{best_cfg['short']} Forecast Price"
    )
    if show_quantile_band:
        ax_prc.fill_between(x_idx, rf_p10, rf_p90, color=best_cfg['color'], alpha=0.14, label=f"{best_cfg['short']} P10-P90 Interval")
    if pd.notnull(rf_p50.iloc[center_idx]):
        ax_prc.annotate(
            f"{best_cfg['short']}: {rf_p50.iloc[center_idx]:,.0f} KRW",
            xy=(center_idx, rf_p50.iloc[center_idx]),
            xytext=(0, 8), textcoords="offset points",
            ha='center', fontsize=8.8, fontweight='bold', color=best_cfg['color']
        )

    ax_prc.set_title(f"[{mkt_en} | {var_en} - {unit_en}] Weekly ({week_range_en}) Price: Actual vs 2021-2025 vs {best_cfg['short']} ({prc_unit_en})", fontsize=11.0, fontweight='bold', pad=10)
    ax_prc.set_xticks(x_idx)
    ax_prc.set_xticklabels(x_labels, fontsize=9.5, fontweight='bold')
    ax_prc.set_ylabel(f"Price ({prc_unit_en})", fontsize=10, fontweight='bold')
    ax_prc.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax_prc.legend(loc='upper right', fontsize=8.5)

    plt.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    st.divider()

    # 3. 4개 모델 개별(2x2) 일주일간 예상 거래량(막대) + 예상 가격(선) 상세 그래프
    st.subheader(f"🔍 2. 각각의 모델별(TFT · XGBoost · LightGBM · Random Forest) 해당 주간({week_range_str}) 실제 vs 예상 거래량 · 가격 상세 그래프")

    ncols_m = 2
    nrows_m = int(np.ceil(len(selected_models) / ncols_m))
    fig_each, axes_each = plt.subplots(nrows_m, ncols_m, figsize=(18.5, nrows_m * 5.1))
    axes_each = np.array(axes_each).reshape(-1)

    for idx_m, m_name in enumerate(selected_models):
        ax1 = axes_each[idx_m]
        m_cfg = MODEL_CONFIG[m_name]
        g_acc = group_acc_dict.get(m_name)
        w_acc = window_acc_dict.get(m_name)

        v_p10 = win_df[f'{m_name}_vol_p10'].fillna(0.0).values
        v_p50 = win_df[f'{m_name}_vol_p50'].fillna(0.0).values
        v_p90 = win_df[f'{m_name}_vol_p90'].fillna(0.0).values

        p_p10 = (win_df[f'{m_name}_prc_p10'] * prc_mult).values
        p_p50 = (win_df[f'{m_name}_prc_p50'] * prc_mult).values
        p_p90 = (win_df[f'{m_name}_prc_p90'] * prc_mult).values

        act_v_sub = win_df['actual_vol_ton'].fillna(0.0).values
        act_p_sub = (win_df['actual_prc_kg'] * prc_mult).values

        ax1.axvspan(center_idx - 0.42, center_idx + 0.42, color='#FFD166', alpha=0.30, zorder=0, label='Selected Date')
        ax1.axvline(center_idx, color='#D62828', linestyle='--', linewidth=1.4, zorder=1)

        if show_hist_21_25 and has_any_actual:
            w_b = 0.24
            ax1.bar(x_idx - w_b, win_df['hist_vol_21_25'].fillna(0.0), width=w_b, color='#8ECAE6', edgecolor='#219EBC', alpha=0.82, label='2021-2025 Vol (ton)')
            ax1.bar(x_idx, act_v_sub, width=w_b, color='#1D3557', alpha=0.90, label='Actual Vol (ton)')
            ax1.bar(x_idx + w_b, v_p50, width=w_b, color=m_cfg['color'], alpha=0.90, label=f"{m_cfg['short']} Forecast Vol (ton)")
            pred_bar_x = x_idx + w_b
            act_bar_x = x_idx
        elif has_any_actual:
            w_b = 0.32
            ax1.bar(x_idx - w_b/2, act_v_sub, width=w_b, color='#1D3557', alpha=0.90, label='Actual Vol (ton)')
            ax1.bar(x_idx + w_b/2, v_p50, width=w_b, color=m_cfg['color'], alpha=0.90, label=f"{m_cfg['short']} Forecast Vol (ton)")
            pred_bar_x = x_idx + w_b/2
            act_bar_x = x_idx - w_b/2
        elif show_hist_21_25:
            w_b = 0.32
            ax1.bar(x_idx - w_b/2, win_df['hist_vol_21_25'].fillna(0.0), width=w_b, color='#8ECAE6', edgecolor='#219EBC', alpha=0.82, label='2021-2025 Vol (ton)')
            ax1.bar(x_idx + w_b/2, v_p50, width=w_b, color=m_cfg['color'], alpha=0.90, label=f"{m_cfg['short']} Forecast Vol (ton)")
            pred_bar_x = x_idx + w_b/2
            act_bar_x = None
        else:
            ax1.bar(x_idx, v_p50, width=0.45, color=m_cfg['color'], alpha=0.90, label=f"{m_cfg['short']} Forecast Vol (ton)")
            pred_bar_x = x_idx
            act_bar_x = None

        if show_quantile_band:
            valid_q = v_p50 > 0
            if valid_q.any():
                yerr_low = np.maximum(0.0, v_p50[valid_q] - v_p10[valid_q])
                yerr_high = np.maximum(0.0, v_p90[valid_q] - v_p50[valid_q])
                ax1.errorbar(pred_bar_x[valid_q], v_p50[valid_q], yerr=[yerr_low, yerr_high], fmt='none', ecolor='#264653', elinewidth=1.3, capsize=3.0, label='Forecast Vol P10-P90')

        for xi, bv in zip(pred_bar_x, v_p50):
            if bv > 0:
                ax1.text(xi, bv * 1.02, f"Pred\n{bv:,.1f}t", ha='center', va='bottom', fontsize=7.3, fontweight='bold', color=m_cfg['color'])

        if has_any_actual and act_bar_x is not None:
            for xi, av in zip(act_bar_x, act_v_sub):
                if av > 0:
                    ax1.text(xi, av * 1.02, f"Act\n{av:,.1f}t", ha='center', va='bottom', fontsize=7.3, fontweight='bold', color='#1D3557')

        ax1.set_title(f"{idx_m + 1}. {m_cfg['label']}", fontsize=11, fontweight='bold', pad=10)
        ax1.set_xticks(x_idx)
        ax1.set_xticklabels(x_labels, fontsize=8.8, fontweight='bold')
        ax1.set_ylabel("Volume (ton/day)", color=m_cfg['color'], fontsize=9.5, fontweight='bold')
        ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.1f}'))
        max_v_local = np.nanmax([
            np.nanmax(v_p90) if np.isfinite(v_p90).any() else 1.0,
            win_df['hist_vol_21_25'].max() if show_hist_21_25 else 1.0,
            np.nanmax(act_v_sub) if has_any_actual else 1.0,
            1.0
        ])
        ax1.set_ylim(0, max_v_local * 1.38)
        ax1.grid(axis='y', linestyle='--', alpha=0.35)
        ax1.grid(axis='x', visible=False)

        ax2 = ax1.twinx()
        if show_hist_21_25:
            ax2.plot(x_idx, win_df['hist_prc_21_25'] * prc_mult, color='#6C757D', linestyle=':', marker='D', markersize=4.8, linewidth=1.6, label=f"2021-2025 Price ({prc_unit_en})")
        if has_any_actual:
            ax2.plot(x_idx, act_p_sub, color='#1D3557', linestyle='-', marker='*', markersize=8.0, linewidth=2.3, zorder=5, label=f"Actual Price ({prc_unit_en})")
            for xi, apv in enumerate(act_p_sub):
                if np.isfinite(apv):
                    ax2.text(xi, apv * 0.965, f"Act {apv:,.0f}", ha='center', va='top', fontsize=7.5, fontweight='bold', color='#1D3557')

        ax2.plot(x_idx, p_p50, color='#D62828', linestyle='--', marker='s', markersize=5.8, linewidth=2.2, zorder=6, label=f"{m_cfg['short']} Forecast Price ({prc_unit_en})")
        if show_quantile_band:
            ax2.fill_between(x_idx, p_p10, p_p90, color='#D62828', alpha=0.13, label='Forecast Price P10-P90')

        for xi, pv in enumerate(p_p50):
            if np.isfinite(pv):
                ax2.text(xi, pv * 1.025, f"Pred {pv:,.0f}", ha='center', va='bottom', fontsize=7.8, fontweight='bold', color='#D62828')

        ax2.set_ylabel(f"Price ({prc_unit_en})", color='#D62828', fontsize=9.5, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#D62828')
        ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax2.grid(False)

        all_p_vals = np.concatenate([
            p_p10[np.isfinite(p_p10)],
            p_p90[np.isfinite(p_p90)],
            act_p_sub[np.isfinite(act_p_sub)] if has_any_actual else np.array([]),
            (win_df['hist_prc_21_25'] * prc_mult).dropna().values if show_hist_21_25 else np.array([])
        ])
        if len(all_p_vals) > 0:
            ax2.set_ylim(max(0, np.min(all_p_vals) * 0.72), np.max(all_p_vals) * 1.25)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=7.2, ncol=2)

    for j in range(len(selected_models), len(axes_each)):
        axes_each[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig_each)
    plt.close(fig_each)

    st.divider()

    # 4. 모델별 가격 및 거래량 예측 정확도 비교 섹션 (그래프 + 상세 요약표)
    st.subheader(f"🎯 3. 모델별 가격 및 거래량 예측 정확도(Accuracy · R² · MAE · MAPE) 비교 분석")

    if len(acc_df) > 0:
        fig_acc, (ax_ap, ax_av) = plt.subplots(1, 2, figsize=(18, 4.8))
        xm = np.arange(len(acc_df))
        has_win_acc = acc_df['_w_p_acc'].notnull().any()
        bw = 0.25 if has_win_acc else 0.34

        # (좌) 모델별 가격 예측 정확도(%) 비교
        if has_win_acc:
            b1 = ax_ap.bar(xm - bw, acc_df['_g_p_acc'], width=bw, color='#2A9D8F', alpha=0.90, label=f"Selected ({var_en} - {unit_en}) Price Acc (%)")
            b2 = ax_ap.bar(xm, acc_df['_w_p_acc'].fillna(0.0), width=bw, color='#E76F51', alpha=0.90, label='Selected Week (Mon-Sat) Price Acc (%)')
            b3 = ax_ap.bar(xm + bw, acc_df['_o_p_acc'], width=bw, color='#457B9D', alpha=0.85, label=f"All {n_total_groups} Units Avg Price Acc (%)")
            for bars_grp in (b1, b2, b3):
                for b in bars_grp:
                    h = b.get_height()
                    if h > 0:
                        ax_ap.text(b.get_x() + b.get_width()/2, h + 0.4, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        else:
            b1 = ax_ap.bar(xm - bw/2, acc_df['_g_p_acc'], width=bw, color='#2A9D8F', alpha=0.90, label=f"Selected ({var_en} - {unit_en}) Price Acc (%)")
            b3 = ax_ap.bar(xm + bw/2, acc_df['_o_p_acc'], width=bw, color='#457B9D', alpha=0.85, label=f"All {n_total_groups} Units Avg Price Acc (%)")
            for bars_grp in (b1, b3):
                for b in bars_grp:
                    h = b.get_height()
                    if h > 0:
                        ax_ap.text(b.get_x() + b.get_width()/2, h + 0.4, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        ax_ap.set_title("1. Price Prediction Accuracy by Model (100 - MAPE, %)", fontsize=12, fontweight='bold', pad=10)
        ax_ap.set_xticks(xm)
        ax_ap.set_xticklabels(acc_df['모델명'], fontsize=10, fontweight='bold')
        ax_ap.set_ylabel("Price Accuracy (%)", fontsize=10, fontweight='bold')
        ax_ap.set_ylim(50, 105)
        ax_ap.legend(loc='lower right', fontsize=8.5)

        # (우) 모델별 거래량 예측 정확도(%) 및 결정계수(R²) 비교
        if has_win_acc:
            bv1 = ax_av.bar(xm - bw, acc_df['_g_v_acc'], width=bw, color='#2A9D8F', alpha=0.90, label=f"Selected ({var_en} - {unit_en}) Vol Acc (100-WMAPE, %)")
            bv2 = ax_av.bar(xm, acc_df['_w_v_acc'].fillna(0.0), width=bw, color='#E76F51', alpha=0.90, label='Selected Week (Mon-Sat) Vol Acc (%)')
            bv3 = ax_av.bar(xm + bw, acc_df['_o_v_r2_pct'], width=bw, color='#8E44AD', alpha=0.85, label=f"All {n_total_groups} Units Vol R2 x 100 (%)")
            for bars_grp in (bv1, bv2, bv3):
                for b in bars_grp:
                    h = b.get_height()
                    if h > 0:
                        ax_av.text(b.get_x() + b.get_width()/2, h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        else:
            bv1 = ax_av.bar(xm - bw/2, acc_df['_g_v_acc'], width=bw, color='#2A9D8F', alpha=0.90, label=f"Selected ({var_en} - {unit_en}) Vol Acc (100-WMAPE, %)")
            bv3 = ax_av.bar(xm + bw/2, acc_df['_o_v_r2_pct'], width=bw, color='#8E44AD', alpha=0.85, label=f"All {n_total_groups} Units Vol R2 x 100 (%)")
            for bars_grp in (bv1, bv3):
                for b in bars_grp:
                    h = b.get_height()
                    if h > 0:
                        ax_av.text(b.get_x() + b.get_width()/2, h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

        ax_av.set_title("2. Volume Prediction Accuracy (100 - WMAPE %) & R2 by Model", fontsize=12, fontweight='bold', pad=10)
        ax_av.set_xticks(xm)
        ax_av.set_xticklabels(acc_df['모델명'], fontsize=10, fontweight='bold')
        ax_av.set_ylabel("Volume Accuracy / R2 (%)", fontsize=10, fontweight='bold')
        ax_av.set_ylim(40, 106)
        ax_av.legend(loc='lower right', fontsize=8.5)

        plt.tight_layout()
        st.pyplot(fig_acc)
        plt.close(fig_acc)

        disp_acc_cols = [c for c in acc_df.columns if not c.startswith('_')]
        st.dataframe(acc_df[disp_acc_cols], use_container_width=True, hide_index=True)

    st.divider()

    # 5. 주간(월~토) 날짜별 · 모델별 상세 예측 데이터 테이블 출력 (실제 가격·거래량 포함)
    st.subheader(f"📋 4. [{selected_market} · {selected_variety} - {selected_unit}] 해당 주간({week_range_str}) 실제 · 예년 · 모델별 예상 가격 및 거래량 상세 비교표")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="📥 해당 주간(월~토) 실제·예년·모델별 예측 결과 CSV 다운로드",
        data=csv_bytes,
        file_name=f"citrus_mon_sat_forecast_{selected_market}_{selected_gid}_{selected_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="btn_download_active"
    )
