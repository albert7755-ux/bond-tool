import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog, curve_fit
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 基礎設定 ---
st.set_page_config(page_title="債券策略大師 Pro (現金流版)", layout="wide")

st.title("🛡️ 債券投資組合策略大師 Pro (現金流版)")
st.markdown("""
針對高資產客戶設計的五大策略：
1. **收益最大化**：追求最高配息。
2. **債券梯**：平均佈局年期，降低風險。
3. **槓鈴策略**：長短債配置。
4. **相對價值**：找出被低估的便宜債券。
5. **現金流組合 (Cash Flow)**：<span style='color:orange'>🔥升級</span> 自訂本金與領息頻率 (月配/季配)，試算退休現金流。
""", unsafe_allow_html=True)

# --- 2. 輔助函式 ---
rating_map = {
    'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
    'A+': 5, 'A': 6, 'A-': 7,
    'BBB+': 8, 'BBB': 9, 'BBB-': 10,
    'BB+': 11, 'BB': 12, 'BB-': 13,
    'B+': 14, 'B': 15, 'B-': 16
}

@st.cache_data
def clean_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, engine='openpyxl')
            
        col_mapping = {}
        for col in df.columns:
            if 'ISIN' in col.upper(): col_mapping[col] = 'ISIN'
            elif '發行' in col or '名稱' in col: col_mapping[col] = 'Name'
            elif 'YTM' in col.upper() or 'YIELD' in col.upper(): col_mapping[col] = 'YTM'
            elif '存續' in col or 'DURATION' in col.upper(): col_mapping[col] = 'Duration'
            elif 'S&P' in col.upper(): col_mapping[col] = 'SP_Rating'
            elif 'FITCH' in col.upper(): col_mapping[col] = 'Fitch_Rating'
            elif '到期' in col or 'MATURITY' in col.upper(): col_mapping[col] = 'Maturity'
        
        df = df.rename(columns=col_mapping)
        
        req_cols = ['ISIN', 'Name', 'YTM', 'Duration']
        if not all(c in df.columns for c in req_cols):
            return None, f"缺少必要欄位，偵測到: {list(df.columns)}"

        df['YTM'] = pd.to_numeric(df['YTM'], errors='coerce')
        df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
        df = df.dropna(subset=['YTM', 'Duration'])
        df = df[df['YTM'] > 0] 

        if 'SP_Rating' in df.columns: df['Rating_Source'] = df['SP_Rating']
        elif 'Fitch_Rating' in df.columns: df['Rating_Source'] = df['Fitch_Rating']
        else: df['Rating_Source'] = 'BBB'
        df['Rating_Source'] = df['Rating_Source'].astype(str).str.strip().str.upper()
        df['Credit_Score'] = df['Rating_Source'].map(rating_map).fillna(10)
        
        # 月份處理
        df['Pay_Month'] = 0
        if 'Maturity' in df.columns:
            try:
                df['Maturity_Dt'] = pd.to_datetime(df['Maturity'], errors='coerce')
                df['Pay_Month'] = df['Maturity_Dt'].dt.month.fillna(0).astype(int)
            except: pass
        
        if df['Pay_Month'].sum() == 0:
            np.random.seed(42)
            df['Pay_Month'] = np.random.randint(1, 7, size=len(df))
            df['Is_Simulated_Month'] = True
        else:
            df['Is_Simulated_Month'] = False
            # 統一歸類到 1-6 (假設半年配)
            df['Pay_Month'] = df['Pay_Month'].apply(lambda x: x if x <= 6 else x - 6)

        return df, None
    except Exception as e:
        return None, str(e)

# --- 3. 策略邏輯 ---

def run_max_yield(df, target_dur, target_score, max_w):
    n = len(df)
    if n == 0: return pd.DataFrame()
    c = -1 * df['YTM'].values
    A_ub = np.array([df['Duration'].values, df['Credit_Score'].values])
    b_ub = np.array([target_dur, target_score])
    A_eq = np.array([np.ones(n)])
    b_eq = np.array([1.0])
    bounds = [(0, max_w) for _ in range(n)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        df['Weight'] = res.x
        return df[df['Weight'] > 0.001].copy()
    return pd.DataFrame()

def run_ladder(df, steps, allow_dup):
    selected = []
    used_issuers = set()
    weight_per_step = 1.0 / len(steps)
    for (min_d, max_d) in steps:
        pool = df[(df['Duration'] >= min_d) & (df['Duration'] < max_d)].sort_values('YTM', ascending=False)
        for idx, row in pool.iterrows():
            if allow_dup or (row['Name'] not in used_issuers):
                best_bond = row.copy()
                best_bond['Weight'] = weight_per_step
                selected.append(best_bond)
                used_issuers.add(row['Name'])
                break
    if selected: return pd.DataFrame(selected)
    return pd.DataFrame()

def run_barbell(df, short_limit, long_limit, long_weight, allow_dup):
    short_pool = df[df['Duration'] <= short_limit].sort_values('YTM', ascending=False)
    long_pool = df[df['Duration'] >= long_limit].sort_values('YTM', ascending=False)
    selected, used_issuers = [], set()
    
    short_picks = []
    for idx, row in short_pool.iterrows():
        if len(short_picks) >= 2: break
        if allow_dup or (row['Name'] not in used_issuers):
            row = row.copy()
            row['Weight'] = (1 - long_weight) / 2 
            short_picks.append(row)
            used_issuers.add(row['Name'])
            
    long_picks = []
    for idx, row in long_pool.iterrows():
        if len(long_picks) >= 2: break
        if allow_dup or (row['Name'] not in used_issuers):
            row = row.copy()
            row['Weight'] = long_weight / 2
            long_picks.append(row)
            used_issuers.add(row['Name'])
    
    final_list = short_picks + long_picks
    if final_list: return pd.DataFrame(final_list)
    return pd.DataFrame()

def fit_yield_curve(x, a, b):
    return a + b * np.log(x)

def run_relative_value(df, allow_dup, top_n, min_dur):
    df_calc = df[df['Duration'] > 0.1].copy()
    if len(df_calc) < 5: return pd.DataFrame(), pd.DataFrame()
    try:
        popt, _ = curve_fit(fit_yield_curve, df_calc['Duration'], df_calc['YTM'])
        df_calc['Fair_YTM'] = fit_yield_curve(df_calc['Duration'], *popt)
        df_calc['Alpha'] = df_calc['YTM'] - df_calc['Fair_YTM']
    except:
        z = np.polyfit(df_calc['Duration'], df_calc['YTM'], 2)
        p = np.poly1d(z)
        df_calc['Fair_YTM'] = p(df_calc['Duration'])
        df_calc['Alpha'] = df_calc['YTM'] - df_calc['Fair_YTM']

    pool = df_calc[df_calc['Duration'] >= min_dur].sort_values('Alpha', ascending=False)
    selected = []
    used_issuers = set()
    weight_per_bond = 1.0 / top_n
    
    count = 0
    for idx, row in pool.iterrows():
        if count >= top_n: break
        if allow_dup or (row['Name'] not in used_issuers):
            bond = row.copy()
            bond['Weight'] = weight_per_bond
            selected.append(bond)
            used_issuers.add(row['Name'])
            count += 1
            
    if selected: return pd.DataFrame(selected), df_calc
    return pd.DataFrame(), df_calc

def run_cash_flow_strategy(df, allow_dup, freq_type):
    """
    現金流策略：
    freq_type: 1=月配(需6檔), 2=雙月配(需3檔), 3=季配(需2檔)
    假設所有債券皆為半年配 (Semi-Annual)
    """
    selected = []
    used_issuers = set()
    
    # 定義需要的月份循環
    if freq_type == "月月配 (12次/年)":
        target_months = [1, 2, 3, 4, 5, 6] # 需要填滿所有月份
    elif freq_type == "雙月配 (6次/年)":
        target_months = [1, 3, 5] # 1,3,5 (會涵蓋 7,9,11)
    else: # "季季配 (4次/年)"
        target_months = [1, 4] # 1,4 (會涵蓋 7,10)
    
    weight_per_bond = 1.0 / len(target_months)
    
    for m in target_months:
        pool = df[df['Pay_Month'] == m].sort_values('YTM', ascending=False)
        found = False
        for idx, row in pool.iterrows():
            if allow_dup or (row['Name'] not in used_issuers):
                bond = row.copy()
                bond['Weight'] = weight_per_bond
                bond['Cycle_Str'] = f"{m}月/{m+6}月"
                selected.append(bond)
                used_issuers.add(row['Name'])
                found = True
                break
    
    if selected: return pd.DataFrame(selected)
    return pd.DataFrame()

# --- 4. 主程式 UI ---
st.sidebar.header("📂 步驟 1: 資料匯入")
uploaded_file = st.sidebar.file_uploader("上傳債券清單", type=['xlsx', 'csv'])

if uploaded_file:
    df_raw, err = clean_data(uploaded_file)
    if err:
        st.error(f"錯誤: {err}")
    else:
        st.sidebar.success(f"已讀取 {len(df_raw)} 檔債券")

        # 黑名單
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚫 黑名單")
        all_issuers = sorted(df_raw['Name'].astype(str).unique())
        excluded_issuers = st.sidebar.multiselect("剔除機構：", options=all_issuers)
        if excluded_issuers:
            df_clean = df_raw[~df_raw['Name'].isin(excluded_issuers)].copy()
        else:
            df_clean = df_raw.copy()
        
        # 策略選擇
        st.sidebar.header("🧠 步驟 2: 選擇策略")
        strategy = st.sidebar.radio(
            "請選擇投資策略：",
            ["收益最大化", "債券梯", "槓鈴策略", "相對價值", "現金流組合 (Cash Flow)"]
        )
        
        # 本金設定 (全域)
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 投資設定")
        investment_amt = st.sidebar.number_input("投資本金 (元)", min_value=10000, value=1000000, step=100000)
        
        allow_dup = True
        if strategy != "收益最大化":
            allow_dup = st.sidebar.checkbox("允許機構重複?", value=True)

        portfolio = pd.DataFrame()
        df_with_alpha = pd.DataFrame() 

        # --- 策略執行 ---
        if strategy == "收益最大化":
            t_dur = st.sidebar.slider("存續期間上限", 2.0, 15.0, 6.0)
            t_cred = rating_map[st.sidebar.select_slider("最低信評", list(rating_map.keys()), 'BBB')]
            max_w = st.sidebar.slider("單檔上限", 0.05, 0.5, 0.2)
            if st.sidebar.button("🚀 計算"):
                portfolio = run_max_yield(df_clean, t_dur, t_cred, max_w)

        elif strategy == "債券梯":
            ladder_type = st.sidebar.selectbox("梯型結構", ["短梯 (1-5年)", "中梯 (3-7年)", "長梯 (5-15年)"])
            ladder_map = {"短梯 (1-5年)": [(1,2),(2,3),(3,4),(4,5)], "中梯 (3-7年)": [(3,4),(4,5),(5,6),(6,7)], "長梯 (5-15年)": [(5,7),(7,10),(10,12),(12,15)]}
            if st.sidebar.button("🚀 計算"):
                portfolio = run_ladder(df_clean, ladder_map[ladder_type], allow_dup)

        elif strategy == "槓鈴策略":
            short_lim = st.sidebar.number_input("短債 < 年", 3.0)
            long_lim = st.sidebar.number_input("長債 > 年", 10.0)
            long_w = st.sidebar.slider("長債佔比", 0.1, 0.9, 0.5)
            if st.sidebar.button("🚀 計算"):
                portfolio = run_barbell(df_clean, short_lim, long_lim, long_w, allow_dup)

        elif strategy == "相對價值":
            min_dur = st.sidebar.number_input("最低年期", 2.0)
            top_n = st.sidebar.slider("挑選幾檔", 3, 10, 5)
            target_rating = st.sidebar.multiselect("篩選信評", sorted(df_clean['Rating_Source'].unique()))
            if st.sidebar.button("🚀 計算"):
                df_t = df_clean[df_clean['Rating_Source'].isin(target_rating)] if target_rating else df_clean
                portfolio, df_with_alpha = run_relative_value(df_t, allow_dup, top_n, min_dur)

        elif strategy == "現金流組合 (Cash Flow)":
            st.sidebar.caption("利用不同月份的半年配債券，構建現金流。")
            freq_type = st.sidebar.selectbox("目標領息頻率", ["月月配 (12次/年)", "雙月配 (6次/年)", "季季配 (4次/年)"])
            
            if df_clean['Is_Simulated_Month'].iloc[0]:
                st.sidebar.warning("⚠️ 警告：使用模擬月份 (請補上到期日欄位)")
            
            if st.sidebar.button("🚀 建立現金流組合"):
                portfolio = run_cash_flow_strategy(df_clean, allow_dup, freq_type)

        # --- 5. 結果顯示 ---
        if not portfolio.empty:
            portfolio['Allocation %'] = (portfolio['Weight'] * 100).round(1)
            # 依照本金計算預估年配息金額
            portfolio['Annual_Coupon_Amt'] = (investment_amt * portfolio['Weight'] * (portfolio['YTM']/100)).round(0)
            
            avg_ytm = (portfolio['YTM'] * portfolio['Weight']).sum()
            total_coupon = portfolio['Annual_Coupon_Amt'].sum()
            
            st.divider()
            k1, k2, k3 = st.columns(3)
            k1.metric("預期年化殖利率", f"{avg_ytm:.2f}%")
            k2.metric("預估年領總息", f"${total_coupon:,.0f}")
            k3.metric("持倉檔數", f"{len(portfolio)} 檔")

            c1, c2 = st.columns([4, 6])
            with c1:
                st.subheader("📋 建議清單")
                cols = ['Name', 'YTM', 'Duration', 'Allocation %', 'Annual_Coupon_Amt']
                if 'Cycle_Str' in portfolio.columns: cols.insert(1, 'Cycle_Str')
                st.dataframe(portfolio[cols], hide_index=True, use_container_width=True, key="res_tab")

            with c2:
                # 現金流圖表 (所有策略通用，但現金流策略最準)
                st.subheader("💰 預估每月入帳金額")
                
                months = list(range(1, 13))
                cash_flow = [0] * 12
                
                for idx, row in portfolio.iterrows():
                    # 假設皆為半年配
                    coupon_amt = row['Annual_Coupon_Amt'] / 2
                    
                    if 'Pay_Month' in row:
                        m = int(row['Pay_Month']) # 1~6
                    else:
                        m = np.random.randint(1,7) # 其他策略若無月份則隨機模擬以示範
                        
                    cash_flow[m-1] += coupon_amt
                    cash_flow[m+5] += coupon_amt
                
                # 美化圖表
                cf_df = pd.DataFrame({'Month': [f"{i}月" for i in months], 'Amount': cash_flow})
                
                # 判斷是否為「現金流策略」，圖表顏色不同
                bar_color = '#2ecc71' if strategy == "現金流組合 (Cash Flow)" else '#3498db'
                
                fig = px.bar(cf_df, x='Month', y='Amount', title=f"本金 ${investment_amt:,.0f} 之每月現金流試算", text_auto=',.0f')
                fig.update_traces(marker_color=bar_color)
                fig.update_layout(yaxis_title="金額 (元)")
                st.plotly_chart(fig, use_container_width=True, key="cf_chart")
                
                # 若是相對價值策略，額外顯示 RV 圖
                if strategy == "相對價值" and not df_with_alpha.empty:
                    st.markdown("---")
                    st.subheader("📊 相對價值曲線")
                    base_data = df_with_alpha
                    x_range = np.linspace(base_data['Duration'].min(), base_data['Duration'].max(), 100)
                    try:
                        popt, _ = curve_fit(fit_yield_curve, base_data['Duration'], base_data['YTM'])
                        y_fair = fit_yield_curve(x_range, *popt)
                    except:
                        z = np.polyfit(base_data['Duration'], base_data['YTM'], 2)
                        p = np.poly1d(z)
                        y_fair = p(x_range)
                    
                    fig_rv = go.Figure()
                    fig_rv.add_trace(go.Scatter(x=base_data['Duration'], y=base_data['YTM'], mode='markers', name='市場', marker=dict(color='lightgrey')))
                    fig_rv.add_trace(go.Scatter(x=x_range, y=y_fair, mode='lines', name='合理價值', line=dict(dash='dash')))
                    fig_rv.add_trace(go.Scatter(x=portfolio['Duration'], y=portfolio['YTM'], mode='markers', name='Buy', marker=dict(color='red', size=15)))
                    st.plotly_chart(fig_rv, use_container_width=True, key="rv_chart_extra")

        elif uploaded_file and st.session_state.get('last_run'):
            st.warning("⚠️ 找不到符合條件的債券。")
else:
    st.info("👈 請先上傳 Excel")
