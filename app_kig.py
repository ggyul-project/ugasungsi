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
    # 전체 기간 거래량 규모가 가장 큰 상위 5개 주요 도매시장 추출
    top5_markets = (
        test_results.groupby(C['MARKET'])[C['VOLUME']].sum()
        .sort_values(ascending=False)
        .head(5)
        .index.astype(str)
        .tolist()
    )
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
    end_future_dt = today_dt + timedelta(days=30)
    
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
    sel_year, sel_month = st.selectbox("연도 / 월 이동", avail_ym, index=ym_idx, format_func=lambda x: f"{x[0]}년 {x[1]}월", key="sb_ym")
    
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
    st.success(f"선택된 기준일: **{base_date}** (향후 7일간 기준)")
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
            fig_p = go.Figure(go.Scatter(x=act_df['거래일자'], y=act_df[f'실제_{box}_낙찰가(원)'], mode='lines+markers', name=f'실제 {box} 낙찰가', line=dict(color='#2ca02c', width=3)))
            fig_p.update_layout(title=f"[실거래 이력] {box} 상자 실제 낙찰가 추이 (원)", height=360)
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
            peak_doy = float(pk_row['피크_연중일수'].to_numpy()[0]) if not pk_row.empty else 350.0
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
                p_yb = float(st_row['요일_기준단가'].to_numpy()[0]) if not st_row.empty else p_ma7
                yoy_row = yoy_stats[(yoy_stats[C['MARKET']] == mkt) & (yoy_stats[C['VARIETY']] == var) & (yoy_stats[C['BOX']] == box) & (yoy_stats['주차'] == wk_num)]
                yoy_v = float(yoy_row['역대주차_평균거래량'].to_numpy()[0]) if not yoy_row.empty else v_ma7
                yoy_p = float(yoy_row['역대주차_평균단가'].to_numpy()[0]) if not yoy_row.empty else p_ma7
                
                dyn_base_v = 0.6 * v_ma7 + 0.4 * yoy_v
                base_vol_log = np.log1p(dyn_base_v)
                row_df = pd.DataFrame([{
                    'target_dt': target_dt, C['MARKET']: mkt, C['VARIETY']: var, C['BOX']: box,
                    '월': target_dt.month, '요일': yoil, '주차': wk_num,
                    '연중일수_sin': np.sin(2 * np.pi * target_dt.dayofyear / 365.0), '연중일수_cos': np.cos(2 * np.pi * target_dt.dayofyear / 365.0),
                    '거래공백일수': gap_days, '휴장직후_개장일': int(gap_days >= 2), '이월누적_가중치': np.log1p(gap_days),
                    '시즌_경과일수': season_day, '실제_출하경과일': actual_elapsed, '피크_거리일수': doy_diff_val, '피크_구간여부': is_peak_zone,
                    '기준거래량_log': base_vol_log, '직전_거래량_log': np.log1p(vol_buf[-1]), '최근3일_거래량_log': np.log1p(v_ma3), '최근7일_거래량_log': np.log1p(v_ma7),
                    '작년동기_거래량_log': np.log1p(yoy_v), '직전_대비_기준거래량비': (vol_buf[-1] + 1) / (dyn_base_v + 1),
                    '최근3일_대비_기준거래량비': (v_ma3 + 1) / (dyn_base_v + 1), '작년동기_대비_최근비': (v_ma7 + 1) / (yoy_v + 1),
                    '거래량_변동계수': v_std7 / (v_ma7 + 1), '직전_단가_배율': prc_buf[-1] / (p_ma7 + 1),
                    '최근3일_단가_배율': p_ma3 / (p_ma7 + 1), '요일_단가_배율': p_yb / (p_ma7 + 1), '작년동기_단가_배율': yoy_p / (p_ma7 + 1)
                }])
                row_df = add_holiday_features(row_df, 'target_dt')
                d_day_val, hol_pre, hol_post, hol_day = int(row_df['명절_D_day'].to_numpy()[0]), int(row_df['명절_성수기_여부'].to_numpy()[0]), int(row_df['명절_직후_여부'].to_numpy()[0]), int(row_df['명절_연휴_여부'].to_numpy()[0])
                if hol_day == 1: mkt_status = '명절/공휴일'
                elif hol_pre == 1: mkt_status = f"명절성수기(D{d_day_val})"
                elif hol_post == 1: mkt_status = f"명절직후(D+{d_day_val})"
                elif is_peak_zone == 1: mkt_status = '출하피크기'
                else: mkt_status = '정상 경매'
                
                row_df[group_cols] = encoder.transform(row_df[group_cols].astype(str))
                pred_v_bu = max(float(np.expm1(base_vol_log + rf_vol_bu.predict(row_df[vol_features]).item())), 0.0)
                td_st = td_yoil_med[(td_yoil_med[C['MARKET']] == mkt) & (td_yoil_med[C['VARIETY']] == var) & (td_yoil_med['요일'] == yoil)]
                td_yb = float(td_st['총물량_요일기준'].to_numpy()[0]) if not td_st.empty else td_ma7
                td_row = pd.DataFrame([{
                    C['MARKET']: mkt, C['VARIETY']: var, '월': target_dt.month, '요일': yoil, '공백일': gap_days, '휴장직후': int(gap_days >= 2),
                    '실제_출하경과일': actual_elapsed, '피크_거리일수': doy_diff_val, '피크_구간여부': is_peak_zone,
                    '명절_D_day': d_day_val, '명절_성수기_여부': hol_pre, '명절_직후_여부': hol_post,
                    '총물량_기준log': np.log1p(td_yb), '직전_총물량log': np.log1p(td_vol_buf[-1]), '최근7일_총물량log': np.log1p(td_ma7)
                }])
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
                fig_p = go.Figure(go.Scatter(x=valid_df['예측일자'], y=valid_df[f'{box}_예상가(원)'], mode='lines+markers', name=f'{box} 예상가', line=dict(color='#d62728', width=3)))
                fig_p.update_layout(title=f"[미래 예측] {box} 상자 예상가 추이 (원)", height=360)
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
                p_1kg, v_kg = float(m_act['kg당_평균가격'].to_numpy().item()), float(m_act[C['VOLUME']].to_numpy().item())
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
                peak_doy = float(pk_row['피크_연중일수'].to_numpy().item()) if not pk_row.empty else 350.0
                doy_diff_val = (base_dt.dayofyear - peak_doy + 182) % 365 - 182
                is_peak_zone = int(abs(doy_diff_val) <= 14)
                v_ma3, p_ma3, v_ma7, p_ma7 = np.mean(vol_buf[-3:]), np.mean(prc_buf[-3:]), np.mean(vol_buf[-7:]), np.mean(prc_buf[-7:])
                v_std7 = np.std(vol_buf[-7:], ddof=1) if len(vol_buf) > 1 else 0.0
                td_ma7 = np.mean(td_vol_buf[-7:])
                st_row = t_stats[(t_stats[C['MARKET']] == m_name) & (t_stats[C['VARIETY']] == var) & (t_stats[C['BOX']] == box) & (t_stats['요일'] == yoil)]
                p_yb = float(st_row['요일_기준단가'].to_numpy().item()) if not st_row.empty else p_ma7
                yoy_row = yoy_stats[(yoy_stats[C['MARKET']] == m_name) & (yoy_stats[C['VARIETY']] == var) & (yoy_stats[C['BOX']] == box) & (yoy_stats['주차'] == wk_num)]
                yoy_v = float(yoy_row['역대주차_평균거래량'].to_numpy().item()) if not yoy_row.empty else v_ma7
                yoy_p = float(yoy_row['역대주차_평균단가'].to_numpy().item()) if not yoy_row.empty else p_ma7
                dyn_base_v = 0.6 * v_ma7 + 0.4 * yoy_v
                base_vol_log = np.log1p(dyn_base_v)
                row_df = pd.DataFrame([{'target_dt': base_dt, C['MARKET']: m_name, C['VARIETY']: var, C['BOX']: box, '월': base_dt.month, '요일': yoil, '주차': wk_num, '연중일수_sin': np.sin(2 * np.pi * base_dt.dayofyear / 365.0), '연중일수_cos': np.cos(2 * np.pi * base_dt.dayofyear / 365.0), '거래공백일수': gap_days, '휴장직후_개장일': int(gap_days >= 2), '이월누적_가중치': np.log1p(gap_days), '시즌_경과일수': season_day, '실제_출하경과일': actual_elapsed, '피크_거리일수': doy_diff_val, '피크_구간여부': is_peak_zone, '기준거래량_log': base_vol_log, '직전_거래량_log': np.log1p(vol_buf[-1]), '최근3일_거래량_log': np.log1p(v_ma3), '최근7일_거래량_log': np.log1p(v_ma7), '작년동기_거래량_log': np.log1p(yoy_v), '직전_대비_기준거래량비': (vol_buf[-1] + 1) / (dyn_base_v + 1), '최근3일_대비_기준거래량비': (v_ma3 + 1) / (dyn_base_v + 1), '작년동기_대비_최근비': (v_ma7 + 1) / (yoy_v + 1), '거래량_변동계수': v_std7 / (v_ma7 + 1), '직전_단가_배율': prc_buf[-1] / (p_ma7 + 1), '최근3일_단가_배율': p_ma3 / (p_ma7 + 1), '요일_단가_배율': p_yb / (p_ma7 + 1), '작년동기_단가_배율': yoy_p / (p_ma7 + 1)}])
                row_df = add_holiday_features(row_df, 'target_dt')
                d_day_val, hol_pre, hol_post = int(row_df['명절_D_day'].to_numpy().item()), int(row_df['명절_성수기_여부'].to_numpy().item()), int(row_df['명절_직후_여부'].to_numpy().item())
                row_df[group_cols] = encoder.transform(row_df[group_cols].astype(str))
                pred_v_bu = max(float(np.expm1(base_vol_log + rf_vol_bu.predict(row_df[vol_features]).item())), 0.0)
                td_st = td_yoil_med[(td_yoil_med[C['MARKET']] == m_name) & (td_yoil_med[C['VARIETY']] == var) & (td_yoil_med['요일'] == yoil)]
                td_yb = float(td_st['총물량_요일기준'].to_numpy().item()) if not td_st.empty else td_ma7
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
        nat_avg_p = comp_mkt_df[f'{box}_상자가(원)'].mean()
        nat_avg_v = comp_mkt_df['거래량(kg)'].mean()
        
        sel_row = comp_mkt_df[comp_mkt_df['도매시장'] == mkt]
        sel_p = float(sel_row[f'{box}_상자가(원)'].to_numpy().item()) if not sel_row.empty else 0.0
        sel_v = float(sel_row['거래량(kg)'].to_numpy().item()) if not sel_row.empty else 0.0
        
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
            
        st.subheader("📋 전국 도매시장별 상세 비교표 (상자가 높은 순)")
        st.dataframe(comp_mkt_df, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.subheader(f"🎯 전체 도매시장별 모델 검증 정확도 ({var} > {box} | 최근 20% 테스트 기준)")
    sub_all_test = test_results[(test_results[C['VARIETY']].astype(str) == var) & (test_results[C['BOX']].astype(str) == box)]

with tab3:
    sub_test = test_results[
        (test_results[C['MARKET']].astype(str) == mkt) &
        (test_results[C['VARIETY']].astype(str) == var) &
        (test_results[C['BOX']].astype(str) == box)
    ].sort_values(C['DATE']).copy()

    train_period_df = bundle.get('train_period_df', pd.DataFrame())
    if not sub_test.empty:
        test_start_dt = sub_test[C['DATE']].min()
        test_end_dt = sub_test[C['DATE']].max()

        t_p_row = train_period_df[
            (train_period_df[C['MARKET']].astype(str) == mkt) &
            (train_period_df[C['VARIETY']].astype(str) == var) &
            (train_period_df[C['BOX']].astype(str) == box)
        ] if not train_period_df.empty else pd.DataFrame()

        if not t_p_row.empty:
            tr_start_str = pd.to_datetime(t_p_row['학습시작일'].to_numpy()).strftime('%Y-%m-%d')
            tr_end_str = pd.to_datetime(t_p_row['학습종료일'].to_numpy()).strftime('%Y-%m-%d')
        else:
            tr_start_str = "2021-01-01"
            tr_end_str = (test_start_dt - timedelta(days=1)).strftime('%Y-%m-%d')

        period_label = f"학습 기간: {tr_start_str} ~ {tr_end_str} (80%) | 테스트 검증 기간: {test_start_dt.strftime('%Y-%m-%d')} ~ {test_end_dt.strftime('%Y-%m-%d')} (20%)"
    else:
        period_label = "테스트 기간 데이터 없음"

    st.subheader(f"🔍 모델 성능 검증 ({mkt} > {var} > {box})")
    st.info(f"📅 **데이터 분할 기준** — {period_label}")

    if len(sub_test) < 3:
        st.warning(f"선택하신 조건은 테스트 검증 표본이 {len(sub_test)}일로 적어 그래프를 생략했습니다.")
    else:
        box_kg = float(box.replace('kg', ''))
        yv, pv = sub_test[C['VOLUME']].values, sub_test['예측_거래량'].values
        yp, pp = sub_test['kg당_평균가격'].values, sub_test['예측_단가'].values
        p_acc = max(0.0, 100.0 - (np.sum(np.abs(yp - pp)) / np.sum(yp) * 100.0))
        v_acc = max(0.0, 100.0 - (np.sum(np.abs(yv - pv)) / np.sum(yv) * 100.0))

        m1, m2, m3 = st.columns(3)
        m1.metric("테스트 단가 정확도", f"{p_acc:.2f}%", f"평균 오차 {np.mean(np.abs(yp - pp)):,.0f}원/kg")
        m2.metric("테스트 거래량 정확도", f"{v_acc:.2f}%", f"평균 오차 {np.mean(np.abs(yv - pv)):,.1f}kg")
        m3.metric("테스트 검증 영업일", f"{len(sub_test):,}일", f"{test_start_dt.strftime('%Y-%m-%d')} ~ {test_end_dt.strftime('%Y-%m-%d')}")

        dates_str = sub_test[C['DATE']].dt.strftime('%Y-%m-%d')
        fig_tp = go.Figure([
            go.Scatter(x=dates_str, y=yp * box_kg, name=f'실제 {box} 상자가', line=dict(color='#2ca02c')),
            go.Scatter(x=dates_str, y=pp * box_kg, name=f'예측 {box} 상자가', line=dict(color='#d62728', dash='dash'))
        ])
        fig_tp.update_layout(title=f"실제 vs 예측 {box} 상자 가격 비교 ({test_start_dt.strftime('%Y-%m-%d')} ~ {test_end_dt.strftime('%Y-%m-%d')})", height=350)
        st.plotly_chart(fig_tp, use_container_width=True)

        fig_tv = go.Figure([
            go.Scatter(x=dates_str, y=yv, name='실제 거래량(kg)', line=dict(color='#1f77b4')),
            go.Scatter(x=dates_str, y=pv, name='예측 거래량(kg)', line=dict(color='#ff7f0e', dash='dash'))
        ])
        fig_tv.update_layout(title=f"실제 vs 예측 거래량 비교 ({test_start_dt.strftime('%Y-%m-%d')} ~ {test_end_dt.strftime('%Y-%m-%d')})", height=350)
        st.plotly_chart(fig_tv, use_container_width=True)
        # ==============================================================================
        # 모델 설계 및 학습 변수·피처 엔지니어링 설명 섹션 추가
        # ==============================================================================
        st.markdown("---")
        st.subheader("🛠️ 모델 설계 및 성능 검증 방법론 (RandomForest)")

        with st.expander("1️⃣ 명절(설·추석)·시장 휴무일·비수확시기 도메인 변수 반영 설명 및 핵심 코드", expanded=True):
            st.markdown("""
            농산물 도매시장 경락 데이터는 일반 공산품과 달리 **명절 선물 수요, 일요일 휴장, 품종별 수확·비수확 시즌**에 따라 가격과 반입량이 급변합니다. 본 모델은 이를 정밀하게 학습하기 위해 아래 3가지 도메인 로직을 적용했습니다.
            * **명절(설날·추석) D-day 및 성수기 효과**: `holidays` 라이브러리를 활용해 명절 전후 `-21일 ~ +10일` 구간을 추출하고, 선물 수요가 집중되는 **명절 성수기(`D-14 ~ D-2`)**와 물량이 급감하는 **명절 직후(`D+1 ~ D+5`)**를 이진 변수로 분리했습니다.
            * **비수확시기(출하 공백) 분리 및 수확 피크기 반영**: 거래 공백이 `21일` 이상 지속되면 비수확기로 판단해 이전 시즌의 이동평균이 다음 시즌 초기에 왜곡을 주지 않도록 `시즌번호`를 분리했습니다. 또한 품종별 **수확 개시일로부터의 경과일(`실제_출하경과일`)**과 **연중 최대 출하 피크일과의 거리(`피크_거리일수`)**를 계산해 시즌 초·중·말의 시세 흐름을 반영했습니다.
            * **시장 휴무일(일요일·공휴일) 및 이월 물량 효과**: 일요일은 경매 휴무일로 자동 제외하며, 휴장 직후 개장일(`거래공백일수 >= 2`)에 누적된 산지 출하 물량이 한꺼번에 유입되는 현상을 `휴장직후_개장일` 및 `이월누적_가중치` 변수로 학습했습니다.
            """)

            st.code("""# [핵심 코드 1] 명절(설날·추석) D-day 및 성수기·직후 파생변수 생성
def add_holiday_features(df_target, date_col):
    dates = pd.to_datetime(df_target[date_col]).values.astype('datetime64[D]')
    diff_matrix = (dates[:, None] - major_holiday_arr[None, :]).astype('timedelta64[D]').astype(int)
    d_day = diff_matrix[np.arange(len(dates)), np.argmin(np.abs(diff_matrix), axis=1)]
    df_target['명절_D_day'] = np.where((d_day >= -21) & (d_day <= 10), d_day, 99)
    df_target['명절_성수기_여부'] = ((d_day >= -14) & (d_day <= -2)).astype(int)
    df_target['명절_직후_여부'] = ((d_day >= 1) & (d_day <= 5)).astype(int)
    df_target['명절_연휴_여부'] = pd.to_datetime(df_target[date_col]).isin(all_holiday_set).astype(int)
    return df_target

# [핵심 코드 2] 비수확시기(21일 이상 출하 공백) 시즌 분리 및 휴무일 직후 이월 가중치
agg_df['거래공백일수'] = agg_df.groupby(group_cols)[COL_DATE].diff().dt.days.fillna(1)
agg_df['시즌번호'] = (agg_df['거래공백일수'] >= 21).groupby([agg_df[c] for c in group_cols]).cumsum()
df_norm['휴장직후_개장일'] = (df_norm['거래공백일수'] >= 2).astype(int)
df_norm['이월누적_가중치'] = np.log1p(df_norm['거래공백일수'])
df_norm['동적_기준거래량'] = 0.6 * df_norm['최근7일_평균거래량'] + 0.4 * df_norm['작년동기_평균거래량']""", language="python")

        with st.expander("2️⃣ RandomForest 모델 학습 구조 및 전체 입력 변수(Feature) 상세", expanded=True):
            st.markdown("""
            * **검증 방식 (시계열 Hold-out 검증)**: 미래 데이터 누수(Data Leakage)를 방지하기 위해 무작위 분할 대신 **과거 80% 기간(`2021-01-01 ~ 2025-10-22`)으로 학습**하고, **가장 최근 20% 기간(`2025-10-23 ~ 2026-11-11`)으로 예측 정확도를 검증**했습니다.
            * **앙상블 예측 아키텍처**:
              1. **거래량 예측 (`0.6 × Bottom-up + 0.4 × Top-down`)**: 개별 규격의 거래량을 예측하는 모델(`rf_vol_bu`, 28개 변수)과 도매시장·품종 전체 총반입량을 예측해 규격별 점유율로 배분하는 모델(`rf_vol_td`, 15개 변수)을 결합해 변동성을 완화했습니다.
              2. **경락 단가 예측 (`rf_price`)**: 최근 7일 이동평균 단가 대비 당일 등락 배율(`target_price_ratio`, 24개 변수)을 예측한 뒤 기준 단가를 곱해 최종 상자 경락가를 산출합니다.
            """)

            c_feat1, c_feat2 = st.columns(2)
            with c_feat1:
                st.markdown("**📦 거래량 예측 모델 학습 변수 (28개)**")
                st.dataframe(pd.DataFrame({
                    '구분': ['시장·품목', '달력·주기', '휴무·비수기', '명절 효과', '최근·작년 수급'],
                    '학습 변수명': [
                        '도매시장, 감귤품종, 포장단위(kg)',
                        '월, 요일, 주차, 연중일수(sin/cos)',
                        '거래공백일수, 휴장직후_개장일, 이월누적_가중치, 시즌_경과일수, 실제_출하경과일, 피크_거리일수, 피크_구간여부',
                        '명절_D_day, 명절_성수기_여부, 명절_직후_여부, 명절_연휴_여부',
                        '기준거래량_log(최근7일+작년동기), 직전/3일/7일/작년동기 거래량_log, 기준대비 비율(3종), 거래량_변동계수'
                    ]
                }), use_container_width=True, hide_index=True)

            with c_feat2:
                st.markdown("**💰 경락 단가 예측 모델 학습 변수 (24개)**")
                st.dataframe(pd.DataFrame({
                    '구분': ['시장·품목', '달력·주기', '휴무·비수기', '명절 효과', '가격·수급 탄력성'],
                    '학습 변수명': [
                        '도매시장, 감귤품종, 포장단위(kg)',
                        '월, 요일, 주차, 연중일수(sin/cos)',
                        '거래공백일수, 휴장직후_개장일, 시즌_경과일수, 실제_출하경과일, 피크_거리일수, 피크_구간여부',
                        '명절_D_day, 명절_성수기_여부, 명절_직후_여부, 명절_연휴_여부',
                        '직전_대비_기준거래량비, 거래량_변동계수, 직전_단가_배율, 최근3일_단가_배율, 요일_단가_배율, 작년동기_단가_배율'
                    ]
                }), use_container_width=True, hide_index=True)

            # 학습된 RandomForest 모델에서 실제 변수 중요도(Feature Importance) Top 10 추출 및 시각화
            imp_p = pd.Series(rf_price.feature_importances_, index=price_features).sort_values(ascending=True).tail(10)
            imp_v = pd.Series(rf_vol_bu.feature_importances_, index=vol_features).sort_values(ascending=True).tail(10)

            g_imp1, g_imp2 = st.columns(2)
            with g_imp1:
                fig_ip = go.Figure(go.Bar(x=imp_p.values, y=imp_p.index, orientation='h', marker_color='#d62728'))
                fig_ip.update_layout(title="단가 예측 모델 핵심 변수 중요도 (Top 10)", xaxis_title="중요도 (Importance)", height=340)
                st.plotly_chart(fig_ip, use_container_width=True)
            with g_imp2:
                fig_iv = go.Figure(go.Bar(x=imp_v.values, y=imp_v.index, orientation='h', marker_color='#1f77b4'))
                fig_iv.update_layout(title="거래량 예측 모델 핵심 변수 중요도 (Top 10)", xaxis_title="중요도 (Importance)", height=340)
                st.plotly_chart(fig_iv, use_container_width=True)
