import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
import calendar
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

st.set_page_config(page_title="감귤 도매시장 수요·시세 예측 시스템", page_icon="🍊", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] button {
    padding: 2px 0px !important;
    min-height: 28px !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_bundle():
    return joblib.load('citrus_rf_bundle.joblib')

bundle = load_bundle()
rf_vol_bu, rf_vol_td, rf_price = bundle['rf_vol_bu'], bundle['rf_vol_td'], bundle['rf_price']
encoder, td_encoder = bundle['encoder'], bundle['td_encoder']
vol_features, td_features, price_features = bundle['vol_features'], bundle['td_features'], bundle['price_features']
group_cols, td_grp = bundle['group_cols'], bundle['td_grp']
t_stats, td_yoil_med, yoy_stats = bundle['t_stats'], bundle['td_yoil_med'], bundle['yoy_stats']
typical_season_stats, test_results = bundle['typical_season_stats'], bundle['test_results']
recent_hist, recent_td, C = bundle['recent_hist'], bundle['recent_td'], bundle['cols']

kr_holidays = holidays.KR(years=range(2020, 2030))
major_holidays = sorted([pd.Timestamp(dt) for dt, name in kr_holidays.items() if any(k in name for k in ['설날', '추석', 'Seollal', 'Chuseok']) and not any(ex in name for ex in ['전날', '다음날', '대체', 'Substitute', 'observed'])])
major_holiday_arr = np.array(major_holidays, dtype='datetime64[D]')
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

st.title("🍊 감귤 주력 4대 품종 실거래 이력 및 미래 시세 예측 시스템")

with st.sidebar:
    st.header("🔍 시장 및 품목 선택")
    top5_markets = test_results.groupby(C['MARKET'])[C['VOLUME']].sum().sort_values(ascending=False).head(5).index.astype(str).tolist()
    if not top5_markets:
        top5_markets = recent_hist.groupby(C['MARKET'])[C['VOLUME']].sum().sort_values(ascending=False).head(5).index.astype(str).tolist()
    mkt = st.selectbox("도매시장 (규모 상위 5개)", top5_markets, index=top5_markets.index('서울가락') if '서울가락' in top5_markets else 0, key="sb_mkt")
    sub_m = recent_hist[recent_hist[C['MARKET']].astype(str) == mkt]

    varieties = sub_m[C['VARIETY']].astype(str).value_counts().index.tolist()
    var = st.selectbox("감귤 품종", varieties, index=varieties.index('조생귤') if '조생귤' in varieties else 0, key="sb_var")
    sub_v = sub_m[sub_m[C['VARIETY']].astype(str) == var]

    boxes = sorted(sub_v[C['BOX']].astype(str).unique().tolist(), key=lambda x: float(x.replace('kg', '')))
    box = st.selectbox("포장 규격", boxes, index=boxes.index('5kg') if '5kg' in boxes else 0, key="sb_box")

    sub_item = sub_v[sub_v[C['BOX']].astype(str) == box]
    actual_dates_sorted = sorted(sub_item[C['DATE']].dt.date.unique().tolist())
    max_global_date = recent_hist[C['DATE']].max().date()
    today_dt = datetime.now(ZoneInfo('Asia/Seoul')).date()
    end_future_dt = today_dt + timedelta(days=7)

    past_valid = set(actual_dates_sorted[:-1]) if len(actual_dates_sorted) > 1 else set()
    future_valid = {max_global_date + timedelta(days=i) for i in range(1, (end_future_dt - max_global_date).days + 1)} if len(actual_dates_sorted) >= 5 else set()
    allowed_dates = past_valid | future_valid
    if not allowed_dates: allowed_dates = {max_global_date}

    if 'selected_base_date' not in st.session_state or st.session_state['selected_base_date'] not in allowed_dates:
        st.session_state['selected_base_date'] = today_dt if today_dt in allowed_dates else max(allowed_dates)

    st.markdown("---")
    st.subheader("📅 기준 날짜 선택 (달력)")
    st.caption("검은 글씨: 선택 가능(실거래일·예측일) / 회색: 거래 없음")

    avail_ym = sorted(list({(d.year, d.month) for d in allowed_dates}))
    cur_ym = (st.session_state['selected_base_date'].year, st.session_state['selected_base_date'].month)
    ym_idx = avail_ym.index(cur_ym) if cur_ym in avail_ym else len(avail_ym) - 1
    sel_year, sel_month = st.selectbox("연도 / 월 이동", avail_ym, index=ym_idx, format_func=lambda x: f"{x}년 {x}월", key="sb_ym")

    yoil_hdr = st.columns(7)
    for idx, y_txt in enumerate(["월", "화", "수", "목", "금", "토", "일"]):
        yoil_hdr[idx].markdown(f"<div style='text-align:center; font-size:12px; font-weight:bold;'>{y_txt}</div>", unsafe_allow_html=True)

    for week in calendar.monthcalendar(sel_year, sel_month):
        cols = st.columns(7)
        for idx, day_num in enumerate(week):
            if day_num == 0: cols[idx].write("")
            else:
                d_obj = date(sel_year, sel_month, day_num)
                is_enabled = d_obj in allowed_dates
                is_selected = (d_obj == st.session_state['selected_base_date'])
                if cols[idx].button(str(day_num), key=f"cal_{sel_year}_{sel_month}_{day_num}", disabled=not is_enabled, type="primary" if is_selected else "secondary", use_container_width=True):
                    st.session_state['selected_base_date'] = d_obj
                    st.rerun()

    base_date = st.session_state['selected_base_date']
    st.success(f"선택된 기준일: **{base_date}**  \n(향후 7일간 기준)")
    horizon = 7

tab1, tab2, tab3 = st.tabs(["📅 실거래 이력 / 미래 시세 예측", "🌐 전국 도매시장 평균과 비교", "🔍 모델 성능 검증"])
with tab1:
    base_dt = pd.to_datetime(base_date)
    end_dt = base_dt + timedelta(days=horizon - 1)
    box_kg = float(box.replace('kg', ''))

    cond = ((recent_hist[C['MARKET']].astype(str) == mkt) & (recent_hist[C['VARIETY']].astype(str) == var) & (recent_hist[C['BOX']].astype(str) == box))
    max_actual_dt = recent_hist[C['DATE']].max()
    past_actual = recent_hist[cond & (recent_hist[C['DATE']] >= base_dt) & (recent_hist[C['DATE']] <= end_dt)].sort_values(C['DATE'])

    if base_dt <= max_actual_dt and not past_actual.empty:
        st.info(f"📌 **{base_date} ~ {past_actual[C['DATE']].max().date()}** 기간의 **실제 경매 거래 내역**입니다.")
        yoil_names = ['월', '화', '수', '목', '금', '토', '일']
        act_df = pd.DataFrame({
            '거래일자': past_actual[C['DATE']].dt.strftime('%Y-%m-%d') + ' (' + past_actual[C['DATE']].dt.dayofweek.map(lambda x: yoil_names[x]) + ')',
            '실제_1kg단가(원)': np.round(past_actual['kg당_평균가격']),
            f'실제_{box}_낙찰가(원)': np.round(past_actual['kg당_평균가격'] * box_kg),
            '실제_거래량(kg)': np.round(past_actual[C['VOLUME']], 1),
            '실제_거래상자수': np.round(past_actual[C['VOLUME']] / box_kg)
        })
        c1, c2, c3 = st.columns(3)
        best_act = act_df.loc[act_df[f'실제_{box}_낙찰가(원)'].idxmax()]
        c1.metric("기간 내 실제 최고가일", best_act['거래일자'], f"{best_act[f'실제_{box}_낙찰가(원)']:,.0f} 원 / {box}")
        c2.metric(f"실제 평균 {box} 상자가", f"{act_df[f'실제_{box}_낙찰가(원)'].mean():,.0f} 원", f"1kg 평균 {act_df['실제_1kg단가(원)'].mean():,.0f} 원")
        c3.metric("일평균 실제 거래량", f"{act_df['실제_거래량(kg)'].mean():,.0f} kg", f"약 {act_df['실제_거래상자수'].mean():,.0f} 상자")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            act_p_vals = act_df[f'실제_{box}_낙찰가(원)']
            act_mean = act_p_vals.mean()
            act_margin = max(3000, act_mean * 0.3, (act_p_vals.max() - act_p_vals.min()) * 1.5)
            act_ymin = max(0, (act_mean - act_margin) // 1000 * 1000)
            act_ymax = ((act_mean + act_margin) // 1000 + 1) * 1000

            fig_p = go.Figure(go.Scatter(x=act_df['거래일자'], y=act_p_vals, mode='lines+markers+text', text=act_p_vals.apply(lambda x: f"{x:,.0f}원"), textposition="top center", name=f'실제 {box} 낙찰가', line=dict(color='#2ca02c', width=3)))
            fig_p.update_layout(title=f"[실거래 이력] {box} 상자 실제 낙찰가 추이 (원)", yaxis=dict(range=[act_ymin, act_ymax], tickformat=",", title="낙찰가 (원)"), height=360)
            st.plotly_chart(fig_p, use_container_width=True)
        with col_g2:
            fig_v = go.Figure(go.Bar(x=act_df['거래일자'], y=act_df['실제_거래량(kg)'], name='실제 거래량(kg)', marker_color='#1f77b4'))
            fig_v.update_layout(title="[실거래 이력] 실제 일별 거래량 추이 (kg)", height=360)
            st.plotly_chart(fig_v, use_container_width=True)
        st.dataframe(act_df, use_container_width=True, hide_index=True)
    else:
        hist_df = recent_hist[cond & (recent_hist[C['DATE']] <= base_dt)].sort_values(C['DATE'])
        td_hist = recent_td[(recent_td[C['MARKET']].astype(str) == mkt) & (recent_td[C['VARIETY']].astype(str) == var) & (recent_td[C['DATE']] <= base_dt)].sort_values(C['DATE'])

        if len(hist_df) < 5:
            st.warning(f"⚠️ 선택한 조건({mkt} > {var} > {box})은 {base_date} 기준 직전 출하 데이터가 부족하거나 비수확시기입니다.")
        else:
            st.success(f"🤖 **{base_date}** 이후 미도래 미래 구간에 대해 **RandomForest AI 모델이 예측한 시세 및 수요**입니다.")
            vol_buf, prc_buf = hist_df[C['VOLUME']].tail(7).tolist(), hist_df['kg당_평균가격'].tail(7).tolist()
            td_vol_buf = td_hist['시장품종_총거래량'].tail(7).tolist() if len(td_hist) >= 3 else [v * 5 for v in vol_buf]
            season_day, last_trade_dt, actual_elapsed = int(hist_df['시즌_경과일수'].to_numpy()[-1]), hist_df[C['DATE']].max(), int(hist_df['실제_출하경과일'].to_numpy()[-1])
            pk_row = typical_season_stats[(typical_season_stats[C['MARKET']] == mkt) & (typical_season_stats[C['VARIETY']] == var)]
            peak_doy = float(pk_row['피크_연중일수'].to_numpy().flat[0]) if not pk_row.empty else 350.0
            yoil_names, forecast_rows = ['월', '화', '수', '목', '금', '토', '일'], []

            for step in range(0, horizon):
                target_dt = base_dt + timedelta(days=step)
                if target_dt <= max_actual_dt: continue
                yoil, wk_num = target_dt.dayofweek, int(target_dt.isocalendar().week)
                d_label = f"{target_dt.strftime('%Y-%m-%d')} ({yoil_names[yoil]})"
                if yoil == 6:
                    forecast_rows.append({'예측일자': d_label, '시장상태': '일요 휴무', '1kg_단가(원)': None, f'{box}_예상가(원)': None, '예측_거래량(kg)': None, '예상_반입상자': None})
                    continue
                season_day += 1
                gap_days = min((target_dt - last_trade_dt).days, 7)
                actual_elapsed += (target_dt - last_trade_dt).days
                doy_diff_val = (target_dt.dayofyear - peak_doy + 182) % 365 - 182
                is_peak_zone = int(abs(doy_diff_val) <= 14)
                v_ma3, p_ma3, v_ma7, p_ma7 = np.mean(vol_buf[-3:]), np.mean(prc_buf[-3:]), np.mean(vol_buf[-7:]), np.mean(prc_buf[-7:])
                v_std7 = np.std(vol_buf[-7:], ddof=1) if len(vol_buf) > 1 else 0.0
                td_ma7 = np.mean(td_vol_buf[-7:])

                st_row = t_stats[(t_stats[C['MARKET']] == mkt) & (t_stats[C['VARIETY']] == var) & (t_stats[C['BOX']] == box) & (t_stats['요일'] == yoil)]
                p_yb = float(st_row['요일_기준단가'].to_numpy().flat[0]) if not st_row.empty else p_ma7
                yoy_row = yoy_stats[(yoy_stats[C['MARKET']] == mkt) & (yoy_stats[C['VARIETY']] == var) & (yoy_stats[C['BOX']] == box) & (yoy_stats['주차'] == wk_num)]
                yoy_v = float(yoy_row['역대주차_평균거래량'].to_numpy().flat[0]) if not yoy_row.empty else v_ma7
                yoy_p = float(yoy_row['역대주차_평균단가'].to_numpy().flat[0]) if not yoy_row.empty else p_ma7

                dyn_base_v = 0.6 * v_ma7 + 0.4 * yoy_v
                base_vol_log = np.log1p(dyn_base_v)
                row_df = pd.DataFrame([{'target_dt': target_dt, C['MARKET']: mkt, C['VARIETY']: var, C['BOX']: box, '월': target_dt.month, '요일': yoil, '주차': wk_num, '연중일수_sin': np.sin(2 * np.pi * target_dt.dayofyear / 365.0), '연중일수_cos': np.cos(2 * np.pi * target_dt.dayofyear / 365.0), '거래공백일수': gap_days, '휴장직후_개장일': int(gap_days >= 2), '이월누적_가중치': np.log1p(gap_days), '시즌_경과일수': season_day, '실제_출하경과일': actual_elapsed, '피크_거리일수': doy_diff_val, '피크_구간여부': is_peak_zone, '기준거래량_log': base_vol_log, '직전_거래량_log': np.log1p(vol_buf[-1]), '최근3일_거래량_log': np.log1p(v_ma3), '최근7일_거래량_log': np.log1p(v_ma7), '작년동기_거래량_log': np.log1p(yoy_v), '직전_대비_기준거래량비': (vol_buf[-1] + 1) / (dyn_base_v + 1), '최근3일_대비_기준거래량비': (v_ma3 + 1) / (dyn_base_v + 1), '작년동기_대비_최근비': (v_ma7 + 1) / (yoy_v + 1), '거래량_변동계수': v_std7 / (v_ma7 + 1), '직전_단가_배율': prc_buf[-1] / (p_ma7 + 1), '최근3일_단가_배율': p_ma3 / (p_ma7 + 1), '요일_단가_배율': p_yb / (p_ma7 + 1), '작년동기_단가_배율': yoy_p / (p_ma7 + 1)}])
                row_df = add_holiday_features(row_df, 'target_dt')
                d_day_val, hol_pre, hol_post, hol_day = int(row_df['명절_D_day'].to_numpy().flat[0]), int(row_df['명절_성수기_여부'].to_numpy().flat[0]), int(row_df['명절_직후_여부'].to_numpy().flat[0]), int(row_df['명절_연휴_여부'].to_numpy().flat[0])
                if hol_day == 1: mkt_status = '명절/공휴일'
                elif hol_pre == 1: mkt_status = f"명절성수기(D{d_day_val})"
                elif hol_post == 1: mkt_status = f"명절직후(D+{d_day_val})"
                elif is_peak_zone == 1: mkt_status = '출하피크기'
                else: mkt_status = '정상 경매'

                row_df[group_cols] = encoder.transform(row_df[group_cols].astype(str))
                pred_v_bu = max(float(np.expm1(base_vol_log + rf_vol_bu.predict(row_df[vol_features]).item())), 0.0)
                td_st = td_yoil_med[(td_yoil_med[C['MARKET']] == mkt) & (td_yoil_med[C['VARIETY']] == var) & (td_yoil_med['요일'] == yoil)]
                td_yb = float(td_st['총물량_요일기준'].to_numpy().flat[0]) if not td_st.empty else td_ma7
                td_row = pd.DataFrame([{C['MARKET']: mkt, C['VARIETY']: var, '월': target_dt.month, '요일': yoil, '공백일': gap_days, '휴장직후': int(gap_days >= 2), '실제_출하경과일': actual_elapsed, '피크_거리일수': doy_diff_val, '피크_구간여부': is_peak_zone, '명절_D_day': d_day_val, '명절_성수기_여부': hol_pre, '명절_직후_여부': hol_post, '총물량_기준log': np.log1p(td_yb), '직전_총물량log': np.log1p(td_vol_buf[-1]), '최근7일_총물량log': np.log1p(td_ma7)}])
                td_row[td_grp] = td_encoder.transform(td_row[td_grp].astype(str))
                pred_td_total = max(float(np.expm1(np.log1p(td_yb) + rf_vol_td.predict(td_row[td_features]).item())), 0.0)
                pred_v = 0.6 * pred_v_bu + 0.4 * (pred_td_total * min(max(v_ma7 / (td_ma7 + 1), 0.0), 1.0))
                pred_p = max(float(rf_price.predict(row_df[price_features]).item()) * p_ma7, 0.0)
                forecast_rows.append({'예측일자': d_label, '시장상태': mkt_status, '1kg_단가(원)': round(pred_p), f'{box}_예상가(원)': round(pred_p * box_kg), '예측_거래량(kg)': round(pred_v), '예상_반입상자': round(pred_v / box_kg)})
                vol_buf.append(pred_v); prc_buf.append(pred_p); td_vol_buf.append(pred_td_total); last_trade_dt = target_dt

            res_df = pd.DataFrame(forecast_rows)
            valid_df = res_df.dropna(subset=[f'{box}_예상가(원)'])
            best_row = valid_df.loc[valid_df[f'{box}_예상가(원)'].idxmax()]
            c1, c2, c3 = st.columns(3)
            c1.metric("최고가 예상일", best_row['예측일자'], f"{best_row[f'{box}_예상가(원)']:,.0f} 원 / {box}")
            c2.metric(f"예상 평균 {box} 상자가", f"{valid_df[f'{box}_예상가(원)'].mean():,.0f} 원", f"1kg 평균 {valid_df['1kg_단가(원)'].mean():,.0f} 원")
            c3.metric("일평균 예상 거래량", f"{valid_df['예측_거래량(kg)'].mean():,.0f} kg", f"약 {valid_df['예상_반입상자'].mean():,.0f} 상자")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                p_vals = valid_df[f'{box}_예상가(원)']
                p_mean = p_vals.mean()
                margin = max(3000, p_mean * 0.3, (p_vals.max() - p_vals.min()) * 1.5)
                y_min = max(0, (p_mean - margin) // 1000 * 1000)
                y_max = ((p_mean + margin) // 1000 + 1) * 1000

                fig_p = go.Figure(go.Scatter(x=valid_df['예측일자'], y=p_vals, mode='lines+markers+text', text=p_vals.apply(lambda x: f"{x:,.0f}원"), textposition="top center", name=f'{box} 예상가', line=dict(color='#d62728', width=3)))
                fig_p.update_layout(title=f"[미래 예측] {box} 상자 예상가 추이 (원)", yaxis=dict(range=[y_min, y_max], tickformat=",", title="예상가 (원)"), height=360)
                st.plotly_chart(fig_p, use_container_width=True)
            with col_g2:
                fig_v = go.Figure(go.Bar(x=valid_df['예측일자'], y=valid_df['예측_거래량(kg)'], name='예측 거래량(kg)', marker_color='#ff7f0e'))
                fig_v.update_layout(title="[미래 예측] 예상 일별 거래량 추이 (kg)", height=360)
                st.plotly_chart(fig_v, use_container_width=True)
            st.dataframe(res_df, use_container_width=True, hide_index=True)
with tab2:
    st.subheader(f"🌐 전국 도매시장 평균 vs [{mkt}] 비교 및 시장별 모델 정확도 ({var} > {box} | 기준일: {base_date})")
    base_dt = pd.to_datetime(base_date)
    box_kg = float(box.replace('kg', ''))
    max_actual_dt = recent_hist[C['DATE']].max()
    target_markets = sorted(recent_hist[(recent_hist[C['VARIETY']].astype(str) == var) & (recent_hist[C['BOX']].astype(str) == box)][C['MARKET']].astype(str).unique().tolist())
    mkt_comp_rows = []

    if base_dt <= max_actual_dt:
        st.info(f"📌 **{base_date}** 기준 전국 도매시장 평균과 **{mkt}**의 **{var} ({box})** 실제 거래 비교입니다.")
        for m_name in target_markets:
            m_act = recent_hist[(recent_hist[C['MARKET']].astype(str) == m_name) & (recent_hist[C['VARIETY']].astype(str) == var) & (recent_hist[C['BOX']].astype(str) == box) & (recent_hist[C['DATE']] == base_dt)]
            if not m_act.empty:
                p_1kg, v_kg = float(m_act['kg당_평균가격'].to_numpy().flat[0]), float(m_act[C['VOLUME']].to_numpy().flat[0])
                mkt_comp_rows.append({'도매시장': m_name, '구분': '실제 거래', '1kg_단가(원)': round(p_1kg), f'{box}_상자가(원)': round(p_1kg * box_kg), '거래량(kg)': round(v_kg, 1), '상자수(박스)': round(v_kg / box_kg)})
    else:
        if base_dt.dayofweek == 6:
            st.warning("⚠️ 선택하신 날짜는 **일요일(도매시장 휴무일)**입니다. 다른 날짜를 선택해 주세요.")
        else:
            st.success(f"🤖 **{base_date}** 기준 전국 도매시장 평균과 **{mkt}**의 **{var} ({box})** AI 예측 비교입니다.")
            yoil, wk_num = base_dt.dayofweek, int(base_dt.isocalendar().week)
            for m_name in target_markets:
                m_hist = recent_hist[(recent_hist[C['MARKET']].astype(str) == m_name) & (recent_hist[C['VARIETY']].astype(str) == var) & (recent_hist[C['BOX']].astype(str) == box) & (recent_hist[C['DATE']] <= base_dt)].sort_values(C['DATE'])
                if len(m_hist) < 5: continue
                m_td = recent_td[(recent_td[C['MARKET']].astype(str) == m_name) & (recent_td[C['VARIETY']].astype(str) == var) & (recent_td[C['DATE']] <= base_dt)].sort_values(C['DATE'])
                vol_buf, prc_buf = m_hist[C['VOLUME']].tail(7).tolist(), m_hist['kg당_평균가격'].tail(7).tolist()
                td_vol_buf = m_td['시장품종_총거래량'].tail(7).tolist() if len(m_td) >= 3 else [v * 5 for v in vol_buf]
                last_dt = m_hist[C['DATE']].max()
                gap_days = min((base_dt - last_dt).days, 7)
                season_day, actual_elapsed = int(m_hist['시즌_경과일수'].to_numpy()[-1]) + 1, int(m_hist['실제_출하경과일'].to_numpy()[-1]) + (base_dt - last_dt).days
                pk_row = typical_season_stats[(typical_season_stats[C['MARKET']] == m_name) & (typical_season_stats[C['VARIETY']] == var)]
                peak_doy = float(pk_row['피크_연중일수'].to_numpy().flat[0]) if not pk_row.empty else 350.0
                doy_diff_val = (base_dt.dayofyear - peak_doy + 182) % 365 - 182
                is_peak_zone = int(abs(doy_diff_val) <= 14)
                v_ma3, p_ma3, v_ma7, p_ma7 = np.mean(vol_buf[-3:]), np.mean(prc_buf[-3:]), np.mean(vol_buf[-7:]), np.mean(prc_buf[-7:])
                v_std7 = np.std(vol_buf[-7:], ddof=1) if len(vol_buf) > 1 else 0.0
                td_ma7 = np.mean(td_vol_buf[-7:])
                st_row = t_stats[(t_stats[C['MARKET']] == m_name) & (t_stats[C['VARIETY']] == var) & (t_stats[C['BOX']] == box) & (t_stats['요일'] == yoil)]
                p_yb = float(st_row['요일_기준단가'].to_numpy().flat[0]) if not st_row.empty else p_ma7
                yoy_row = yoy_stats[(yoy_stats[C['MARKET']] == m_name) & (yoy_stats[C['VARIETY']] == var) & (yoy_stats[C['BOX']] == box) & (yoy_stats['주차'] == wk_num)]
                yoy_v = float(yoy_row['역대주차_평균거래량'].to_numpy().flat[0]) if not yoy_row.empty else v_ma7
                yoy_p = float(yoy_row['역대주차_평균단가'].to_numpy().flat[0]) if not yoy_row.empty else p_ma7
                dyn_base_v = 0.6 * v_ma7 + 0.4 * yoy_v
                base_vol_log = np.log1p(dyn_base_v)
                row_df = pd.DataFrame([{'target_dt': base_dt, C['MARKET']: m_name, C['VARIETY']: var, C['BOX']: box, '월': base_dt.month, '요일': yoil, '주차': wk_num, '연중일수_sin': np.sin(2 * np.pi * base_dt.dayofyear / 365.0), '연중일수_cos': np.cos(2 * np.pi * base_dt.dayofyear / 365.0), '거래공백일수': gap_days, '휴장직후_개장일': int(gap_days >= 2), '이월누적_가중치': np.log1p(gap_days), '시즌_경과일수': season_day, '실제_출하경과일': actual_elapsed, '피크_거리일수': doy_diff_val, '피크_구간여부': is_peak_zone, '기준거래량_log': base_vol_log, '직전_거래량_log': np.log1p(vol_buf[-1]), '최근3일_거래량_log': np.log1p(v_ma3), '최근7일_거래량_log': np.log1p(v_ma7), '작년동기_거래량_log': np.log1p(yoy_v), '직전_대비_기준거래량비': (vol_buf[-1] + 1) / (dyn_base_v + 1), '최근3일_대비_기준거래량비': (v_ma3 + 1) / (dyn_base_v + 1), '작년동기_대비_최근비': (v_ma7 + 1) / (yoy_v + 1), '거래량_변동계수': v_std7 / (v_ma7 + 1), '직전_단가_배율': prc_buf[-1] / (p_ma7 + 1), '최근3일_단가_배율': p_ma3 / (p_ma7 + 1), '요일_단가_배율': p_yb / (p_ma7 + 1), '작년동기_단가_배율': yoy_p / (p_ma7 + 1)}])
                row_df = add_holiday_features(row_df, 'target_dt')
                d_day_val, hol_pre, hol_post = int(row_df['명절_D_day'].to_numpy().flat[0]), int(row_df['명절_성수기_여부'].to_numpy().flat[0]), int(row_df['명절_직후_여부'].to_numpy().flat[0])
                row_df[group_cols] = encoder.transform(row_df[group_cols].astype(str))
                pred_v_bu = max(float(np.expm1(base_vol_log + rf_vol_bu.predict(row_df[vol_features]).item())), 0.0)
                td_st = td_yoil_med[(td_yoil_med[C['MARKET']] == m_name) & (td_yoil_med[C['VARIETY']] == var) & (td_yoil_med['요일'] == yoil)]
                td_yb = float(td_st['총물량_요일기준'].to_numpy().flat[0]) if not td_st.empty else td_ma7
                td_row = pd.DataFrame([{C['MARKET']: m_name, C['VARIETY']: var, '월': base_dt.month, '요일': yoil, '공백일': gap_days, '휴장직후': int(gap_days >= 2), '실제_출하경과일': actual_elapsed, '피크_거리일수': doy_diff_val, '피크_구간여부': is_peak_zone, '명절_D_day': d_day_val, '명절_성수기_여부': hol_pre, '명절_직후_여부': hol_post, '총물량_기준log': np.log1p(td_yb), '직전_총물량log': np.log1p(td_vol_buf[-1]), '최근7일_총물량log': np.log1p(td_ma7)}])
                td_row[td_grp] = td_encoder.transform(td_row[td_grp].astype(str))
                pred_td_total = max(float(np.expm1(np.log1p(td_yb) + rf_vol_td.predict(td_row[td_features]).item())), 0.0)
                pred_v = 0.6 * pred_v_bu + 0.4 * (pred_td_total * min(max(v_ma7 / (td_ma7 + 1), 0.0), 1.0))
                pred_p = max(float(rf_price.predict(row_df[price_features]).item()) * p_ma7, 0.0)
                mkt_comp_rows.append({'도매시장': m_name, '구분': 'AI 예측', '1kg_단가(원)': round(pred_p), f'{box}_상자가(원)': round(pred_p * box_kg), '거래량(kg)': round(pred_v), '상자수(박스)': round(pred_v / box_kg)})
    if not mkt_comp_rows:
        st.warning("선택하신 날짜와 품목 조건에 해당하는 전국 도매시장 데이터가 없습니다.")
    else:
        comp_mkt_df = pd.DataFrame(mkt_comp_rows).sort_values(by=f'{box}_상자가(원)', ascending=False).reset_index(drop=True)
        nat_avg_p, nat_avg_v = comp_mkt_df[f'{box}_상자가(원)'].mean(), comp_mkt_df['거래량(kg)'].mean()
        sel_row = comp_mkt_df[comp_mkt_df['도매시장'] == mkt]
        sel_p = float(sel_row[f'{box}_상자가(원)'].to_numpy().flat[0]) if not sel_row.empty else 0.0
        sel_v = float(sel_row['거래량(kg)'].to_numpy().flat[0]) if not sel_row.empty else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"전국 평균 {box} 상자가", f"{nat_avg_p:,.0f} 원", f"전국 {len(comp_mkt_df)}개 시장 평균")
        k2.metric(f"[{mkt}] {box} 상자가", f"{sel_p:,.0f} 원" if sel_p > 0 else "데이터 없음", f"전국 평균 대비 {sel_p - nat_avg_p:+,.0f} 원" if sel_p > 0 else None)
        k3.metric("전국 시장당 평균 거래량", f"{nat_avg_v:,.0f} kg", f"총 합계 {comp_mkt_df['거래량(kg)'].sum():,.0f} kg")
        k4.metric(f"[{mkt}] 거래량", f"{sel_v:,.0f} kg" if sel_v > 0 else "데이터 없음", f"전국 평균 대비 {sel_v - nat_avg_v:+,.0f} kg" if sel_v > 0 else None)

        colors_p = ['#d62728' if m == mkt else '#b0bec5' for m in comp_mkt_df['도매시장']]
        colors_v = ['#1f77b4' if m == mkt else '#b0bec5' for m in comp_mkt_df['도매시장']]
        g1, g2 = st.columns(2)
        with g1:
            fig_mp = go.Figure(go.Bar(x=comp_mkt_df['도매시장'], y=comp_mkt_df[f'{box}_상자가(원)'], marker_color=colors_p, text=comp_mkt_df[f'{box}_상자가(원)'].apply(lambda x: f"{x:,.0f}"), textposition='auto'))
            fig_mp.add_hline(y=nat_avg_p, line_dash="dash", line_color="#2ca02c", annotation_text=f"전국 평균 ({nat_avg_p:,.0f}원)", annotation_position="top right")
            fig_mp.update_layout(title=f"전국 도매시장 vs [{mkt}] {box} 상자 가격 비교 (원)", height=380)
            st.plotly_chart(fig_mp, use_container_width=True)
        with g2:
            fig_mv = go.Figure(go.Bar(x=comp_mkt_df['도매시장'], y=comp_mkt_df['거래량(kg)'], marker_color=colors_v, text=comp_mkt_df['거래량(kg)'].apply(lambda x: f"{x:,.0f}"), textposition='auto'))
            fig_mv.add_hline(y=nat_avg_v, line_dash="dash", line_color="#2ca02c", annotation_text=f"전국 평균 ({nat_avg_v:,.0f}kg)", annotation_position="top right")
            fig_mv.update_layout(title=f"전국 도매시장 vs [{mkt}] 거래량 비교 (kg)", height=380)
            st.plotly_chart(fig_mv, use_container_width=True)
        st.dataframe(comp_mkt_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader(f"🎯 전체 도매시장별 모델 검증 정확도 ({var} > {box} | 최근 테스트 기준)")
    sub_all_test = test_results[(test_results[C['VARIETY']].astype(str) == var) & (test_results[C['BOX']].astype(str) == box)]
    if not sub_all_test.empty:
        tot_yp, tot_pp = sub_all_test['kg당_평균가격'].values, sub_all_test['예측_단가'].values
        tot_yv, tot_pv = sub_all_test[C['VOLUME']].values, sub_all_test['예측_거래량'].values
        all_p_acc = max(0.0, 100.0 - (np.sum(np.abs(tot_yp - tot_pp)) / np.sum(tot_yp) * 100.0))
        all_v_acc = max(0.0, 100.0 - (np.sum(np.abs(tot_yv - tot_pv)) / np.sum(tot_yv) * 100.0))
        a1, a2, a3 = st.columns(3)
        a1.metric("전국 전체 단가 예측 정확도", f"{all_p_acc:.2f}%", f"평균 오차 {np.mean(np.abs(tot_yp - tot_pp)):,.0f} 원/kg")
        a2.metric("전국 전체 거래량 예측 정확도", f"{all_v_acc:.2f}%", f"평균 오차 {np.mean(np.abs(tot_yv - tot_pv)):,.1f} kg")
        a3.metric("전체 테스트 표본 수", f"{len(sub_all_test):,} 건", f"{sub_all_test[C['MARKET']].nunique()}개 도매시장")

        acc_rows = []
        for m_name, m_sub in sub_all_test.groupby(C['MARKET']):
            if len(m_sub) < 3: continue
            yp, pp, yv, pv = m_sub['kg당_평균가격'].values, m_sub['예측_단가'].values, m_sub[C['VOLUME']].values, m_sub['예측_거래량'].values
            acc_rows.append({'도매시장': m_name, '테스트_영업일수': len(m_sub), '단가_정확도(%)': round(max(0.0, 100.0 - (np.sum(np.abs(yp - pp)) / np.sum(yp) * 100.0)), 2), '단가_평균오차(원/kg)': round(np.mean(np.abs(yp - pp))), '거래량_정확도(%)': round(max(0.0, 100.0 - (np.sum(np.abs(yv - pv)) / np.sum(yv) * 100.0)), 2), '거래량_평균오차(kg)': round(np.mean(np.abs(yv - pv)), 1)})
        if acc_rows:
            st.dataframe(pd.DataFrame(acc_rows).sort_values(by='단가_정확도(%)', ascending=False).reset_index(drop=True), use_container_width=True, hide_index=True)
with tab3:
    sub_val = bundle.get('val_results', pd.DataFrame())
    if not sub_val.empty:
        sub_val = sub_val[(sub_val[C['MARKET']].astype(str) == mkt) & (sub_val[C['VARIETY']].astype(str) == var) & (sub_val[C['BOX']].astype(str) == box)].sort_values(C['DATE']).copy()
    sub_test = test_results[(test_results[C['MARKET']].astype(str) == mkt) & (test_results[C['VARIETY']].astype(str) == var) & (test_results[C['BOX']].astype(str) == box)].sort_values(C['DATE']).copy()

    split_dates_df = bundle.get('split_dates_df', pd.DataFrame())
    sp_row = split_dates_df[(split_dates_df[C['MARKET']].astype(str) == mkt) & (split_dates_df[C['VARIETY']].astype(str) == var) & (split_dates_df[C['BOX']].astype(str) == box)] if not split_dates_df.empty else pd.DataFrame()
    if not sp_row.empty:
        tr_s, tr_e = pd.to_datetime(sp_row['Train_시작일'].to_numpy().flat[0]).strftime('%Y-%m-%d'), pd.to_datetime(sp_row['Train_종료일'].to_numpy().flat[0]).strftime('%Y-%m-%d')
        va_s, va_e = pd.to_datetime(sp_row['Val_시작일'].to_numpy().flat[0]).strftime('%Y-%m-%d'), pd.to_datetime(sp_row['Val_종료일'].to_numpy().flat[0]).strftime('%Y-%m-%d')
        te_s, te_e = pd.to_datetime(sp_row['Test_시작일'].to_numpy().flat[0]).strftime('%Y-%m-%d'), pd.to_datetime(sp_row['Test_종료일'].to_numpy().flat[0]).strftime('%Y-%m-%d')
        period_msg = f"**Train (70%)**: `{tr_s} ~ {tr_e}`  |  **Validation (15%, 튜닝)**: `{va_s} ~ {va_e}`  |  **Test (15%, 최종평가)**: `{te_s} ~ {te_e}`"
    else:
        period_msg = f"학습 기간: ~ {(sub_test[C['DATE']].min() - timedelta(days=1)).strftime('%Y-%m-%d')} | 테스트 기간: {sub_test[C['DATE']].min().strftime('%Y-%m-%d')} ~ {sub_test[C['DATE']].max().strftime('%Y-%m-%d')}" if not sub_test.empty else "데이터 없음"

    st.subheader(f"🔍 모델 성능 검증 ({mkt} > {var} > {box})")
    st.info(f"📅 **시계열 시점 기준 분할 기간** — {period_msg}")

    if len(sub_test) < 3:
        st.warning(f"선택하신 조건은 Test 검증 표본이 {len(sub_test)}일로 적어 그래프를 생략했습니다.")
    else:
        box_kg = float(box.replace('kg', ''))
        if not sub_val.empty:
            vy_p, vp_p, vy_v, vp_v = sub_val['kg당_평균가격'].values, sub_val['예측_단가'].values, sub_val[C['VOLUME']].values, sub_val['예측_거래량'].values
            val_p_acc, val_v_acc = max(0.0, 100.0 - (np.sum(np.abs(vy_p - vp_p)) / np.sum(vy_p) * 100.0)), max(0.0, 100.0 - (np.sum(np.abs(vy_v - vp_v)) / np.sum(vy_v) * 100.0))
        else:
            val_p_acc, val_v_acc = 0.0, 0.0

        yv, pv, yp, pp = sub_test[C['VOLUME']].values, sub_test['예측_거래량'].values, sub_test['kg당_평균가격'].values, sub_test['예측_단가'].values
        p_acc, v_acc = max(0.0, 100.0 - (np.sum(np.abs(yp - pp)) / np.sum(yp) * 100.0)), max(0.0, 100.0 - (np.sum(np.abs(yv - pv)) / np.sum(yv) * 100.0))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Validation 단가 정확도", f"{val_p_acc:.2f}%", f"검증 {len(sub_val):,}일")
        m2.metric("최종 Test 단가 정확도", f"{p_acc:.2f}%", f"평균 오차 {np.mean(np.abs(yp - pp)):,.0f}원/kg")
        m3.metric("Validation 거래량 정확도", f"{val_v_acc:.2f}%", f"검증 {len(sub_val):,}일")
        m4.metric("최종 Test 거래량 정확도", f"{v_acc:.2f}%", f"평균 오차 {np.mean(np.abs(yv - pv)):,.1f}kg")

        dates_str = sub_test[C['DATE']].dt.strftime('%Y-%m-%d')
        fig_tp = go.Figure([go.Scatter(x=dates_str, y=yp * box_kg, name=f'실제 {box} 상자가', line=dict(color='#2ca02c')), go.Scatter(x=dates_str, y=pp * box_kg, name=f'예측 {box} 상자가', line=dict(color='#d62728', dash='dash'))])
        fig_tp.update_layout(title=f"[최종 Test 구간] 실제 vs 예측 {box} 상자 가격 비교", height=350)
        st.plotly_chart(fig_tp, use_container_width=True)

        fig_tv = go.Figure([go.Scatter(x=dates_str, y=yv, name='실제 거래량(kg)', line=dict(color='#1f77b4')), go.Scatter(x=dates_str, y=pv, name='예측 거래량(kg)', line=dict(color='#ff7f0e', dash='dash'))])
        fig_tv.update_layout(title="[최종 Test 구간] 실제 vs 예측 거래량 비교", height=350)
        st.plotly_chart(fig_tv, use_container_width=True)

        st.markdown("---")
        st.subheader("🛠️ 모델 설계 및 성능 검증 방법론 (RandomForest)")
        with st.expander("1️⃣ 명절(설·추석)·시장 휴무일·비수확시기 도메인 변수 반영 설명 및 핵심 코드", expanded=True):
            st.markdown("""
            * **명절(설날·추석) D-day 및 성수기 효과**: 명절 전후 `-21일 ~ +10일` 구간을 추출하고, 선물 수요가 집중되는 **명절 성수기(`D-14 ~ D-2`)**와 물량이 급감하는 **명절 직후(`D+1 ~ D+5`)**를 파생변수로 분리했습니다.
            * **비수확시기(출하 공백) 분리 및 수확 피크기 반영**: 거래 공백이 `21일` 이상 지속되면 비수확기로 판단해 시즌을 분리(`시즌번호`)하고, 품종별 수확 개시 경과일 및 출하 피크일과의 거리(`피크_거리일수`)를 반영했습니다.
            * **시장 휴무일 및 이월 물량 효과**: 일요일은 경매 휴무일로 제외하며, 휴장 직후 개장일(`거래공백일수 >= 2`)에 누적 물량이 유입되는 효과를 `휴장직후_개장일` 및 `이월누적_가중치`로 학습했습니다.
            """)
            st.code("""# 명절 D-day 및 비수확시기·휴무일 이월 가중치 핵심 로직
df_target['명절_성수기_여부'] = ((d_day >= -14) & (d_day <= -2)).astype(int)
agg_df['시즌번호'] = (agg_df['거래공백일수'] >= 21).groupby([agg_df[c] for c in group_cols]).cumsum()
df_norm['휴장직후_개장일'] = (df_norm['거래공백일수'] >= 2).astype(int)
df_norm['동적_기준거래량'] = 0.6 * df_norm['최근7일_평균거래량'] + 0.4 * df_norm['작년동기_평균거래량']""", language="python")

        with st.expander("2️⃣ RandomForest 모델 학습·튜닝 구조 및 변수 중요도(Top 10)", expanded=True):
            st.markdown("""
            * **시점 기준 3분할 (`Train 70% / Validation 15% / Test 15%`)**: 과거 70% 데이터로 학습하고, 중간 15% Validation 셋으로 최적 트리 깊이(`max_depth`)를 튜닝한 뒤, 가장 최근 15% Test 셋에서 최종 정확도를 검증했습니다.
            * **앙상블 예측 구조**: 거래량 예측 모델 2종(`0.6 × Bottom-up + 0.4 × Top-down`)과 경락 단가 등락 배율 예측 모델(`rf_price`)을 결합했습니다.
            """)
            imp_p = pd.Series(rf_price.feature_importances_, index=price_features).sort_values(ascending=True).tail(10)
            imp_v = pd.Series(rf_vol_bu.feature_importances_, index=vol_features).sort_values(ascending=True).tail(10)
            g_imp1, g_imp2 = st.columns(2)
            with g_imp1:
                fig_ip = go.Figure(go.Bar(x=imp_p.values, y=imp_p.index, orientation='h', marker_color='#d62728'))
                fig_ip.update_layout(title="단가 예측 모델 핵심 변수 중요도 (Top 10)", height=320)
                st.plotly_chart(fig_ip, use_container_width=True)
            with g_imp2:
                fig_iv = go.Figure(go.Bar(x=imp_v.values, y=imp_v.index, orientation='h', marker_color='#1f77b4'))
                fig_iv.update_layout(title="거래량 예측 모델 핵심 변수 중요도 (Top 10)", height=320)
                st.plotly_chart(fig_iv, use_container_width=True)
