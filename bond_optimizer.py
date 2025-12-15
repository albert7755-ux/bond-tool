import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog, curve_fit
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 基礎設定 ---
st.set_page_config(page_title="債券策略大師 Pro (Quant版)", layout="wide")

st.title("🛡️ 債券投資組合策略大師 Pro (Quant版)")
st.markdown("""
針對高資產客戶設計的策略 (含學理相對價值分析)：
1. **收益最大化 (Max Yield)**：在風險限制下追求最高配息。
2. **債券梯 (Ladder)**：平均佈局不同年期，打造穩定現金流。
3. **槓鈴策略 (Barbell)**：長短債配置，兼顧流動性與資本利得。
4. **相對價值 (Relative Value)**：<span style='color:red'>🔥Quant 模型</span>，透過殖利率曲線回歸，找出被市場低估的「超額報酬」債券。
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
        
        return df, None
    except Exception as e:
        return None, str(e)

# --- 3. 策略邏輯核心 ---

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

# 相對價值模型
def fit_yield_curve(x, a, b):
    # 使用對數函數擬合: YTM = a + b * ln(Duration)
    return a + b * np.log(x)

def run_relative_value(df, allow_dup, top_n, min_dur):
    """相對價值策略：加入 min_dur 篩選"""
    
    # 先做初步篩選
    df_calc = df[df['Duration'] > 0.1].copy()
    if len(df_calc) < 5: return pd.DataFrame(), pd.DataFrame()

    # 1. 計算全市場的回歸曲線 (用所有資料算才準)
    try:
        popt, _ = curve_fit(fit_yield_curve, df_calc['Duration'], df_calc['YTM'])
        df_calc['Fair_YTM'] = fit_yield_curve(df_calc['Duration'], *popt)
        df_calc['Alpha'] = df_calc['YTM'] - df_calc['Fair_YTM']
    except:
        z = np.polyfit(df_calc['Duration'], df_calc['YTM'], 2)
        p = np.poly1d(z)
        df_calc['Fair_YTM'] = p(df_calc['Duration'])
        df_calc['Alpha'] = df_calc['YTM'] - df_calc['Fair_YTM']

    # 2. 篩選：只從符合「最低年期」的債券中挑選 Alpha 最高的
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
            
    if selected:
        return pd.DataFrame(selected), df_calc
    return pd.DataFrame(), df_calc


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
        st.sidebar.subheader("🚫 黑名單管理")
        all_issuers = sorted(df_raw['Name'].astype(str).unique())
        excluded_issuers = st.sidebar.multiselect("剔除發行機構：", options=all_issuers)
        if excluded_issuers:
            df_clean = df_raw[~df_raw['Name'].isin(excluded_issuers)].copy()
        else:
            df_clean = df_raw.copy()
        
        # 策略選擇
        st.sidebar.header("🧠 步驟 2: 選擇策略")
        strategy = st.sidebar.radio(
            "請選擇投資策略：",
            ["收益最大化 (Max Yield)", "債券梯 (Ladder)", "槓鈴策略 (Barbell)", "相對價值 (Relative Value)"]
        )
        
        # 共用風控
        allow_dup = True
        if strategy != "收益最大化 (Max Yield)":
            st.sidebar.markdown("---")
            st.sidebar.subheader("🛡️ 集中度風控")
            allow_dup = st.sidebar.checkbox("允許發行機構重複?", value=True)

        portfolio = pd.DataFrame()
        df_with_alpha = pd.DataFrame() 

        # --- 策略執行區 ---
        if strategy == "收益最大化 (Max Yield)":
            st.sidebar.caption("說明：透過演算法算出最高殖利率組合。")
            t_dur = st.sidebar.slider("存續期間上限", 2.0, 15.0, 6.0)
            t_cred_label = st.sidebar.select_slider("最低信評要求", options=list(rating_map.keys()), value='BBB')
            t_cred = rating_map[t_cred_label]
            max_w = st.sidebar.slider("單檔上限", 0.05, 0.5, 0.2)
            if st.sidebar.button("🚀 計算最佳配置"):
                portfolio = run_max_yield(df_clean, t_dur, t_cred, max_w)

        elif strategy == "債券梯 (Ladder)":
            st.sidebar.caption("說明：資金平均分配在不同年期。")
            ladder_options = {
                "短梯 (1-5年)": [(1,2), (2,3), (3,4), (4,5)],
                "中梯 (3-7年)": [(3,4), (4,5), (5,6), (6,7)],
                "長梯 (5-15年)": [(5,7), (7,10), (10,12), (12,15)]
            }
            ladder_type = st.sidebar.selectbox("選擇梯型結構", list(ladder_options.keys()))
            if st.sidebar.button("🚀 建立債券梯"):
                portfolio = run_ladder(df_clean, ladder_options[ladder_type], allow_dup)

        elif strategy == "槓鈴策略 (Barbell)":
            st.sidebar.caption("說明：集中投資極短與極長債。")
            col_s, col_l = st.sidebar.columns(2)
            short_lim = col_s.number_input("短債定義 (年以下)", value=3.0)
            long_lim = col_l.number_input("長債定義 (年以上)", value=10.0)
            long_w = st.sidebar.slider("長債資金佔比", 0.1, 0.9, 0.5)
            if st.sidebar.button("🚀 建立槓鈴組合"):
                portfolio = run_barbell(df_clean, short_lim, long_lim, long_w, allow_dup)

        elif strategy == "相對價值 (Relative Value)":
            st.sidebar.caption("說明：尋找位於殖利率曲線上方(被低估)的債券。")
            
            # 新增：最低存續期間篩選
            min_dur = st.sidebar.number_input("最低存續期間 (年以上)", min_value=0.0, value=2.0, step=0.5)
            
            top_n = st.sidebar.slider("挑選 Alpha 最高的幾檔?", 3, 10, 5)
            
            st.sidebar.info("💡 建議先篩選特定信評等級 (例如只看 BBB)，模型會更準確。")
            target_rating_group = st.sidebar.multiselect(
                "篩選信評 (可複選, 留空則全選)", 
                options=sorted(df_clean['Rating_Source'].unique()),
                default=[]
            )
            
            if st.sidebar.button("🚀 尋找被低估債券"):
                df_target = df_clean.copy()
                if target_rating_group:
                    df_target = df_target[df_target['Rating_Source'].isin(target_rating_group)]
                
                # 傳入 min_dur
                portfolio, df_with_alpha = run_relative_value(df_target, allow_dup, top_n, min_dur)

        # --- 5. 結果顯示區 ---
        if not portfolio.empty:
            portfolio['Allocation %'] = (portfolio['Weight'] * 100).round(1)
            avg_ytm = (portfolio['YTM'] * portfolio['Weight']).sum()
            avg_dur = (portfolio['Duration'] * portfolio['Weight']).sum()
            unique_issuers = portfolio['Name'].nunique()
            
            st.divider()
            k1, k2, k3 = st.columns(3)
            k1.metric("預期年化殖利率 (YTM)", f"{avg_ytm:.2f}%")
            k2.metric("平均存續期間", f"{avg_dur:.2f} 年")
            k3.metric("發行機構數", f"{unique_issuers} 家", delta="集中度檢查")
            
            c1, c2 = st.columns([4, 6])
            
            with c1:
                st.subheader("📋 建議清單")
                show_cols = ['Name', 'ISIN', 'Rating_Source', 'YTM', 'Duration', 'Allocation %']
                if 'Alpha' in portfolio.columns: show_cols.insert(4, 'Alpha')
                
                st.dataframe(
                    portfolio[show_cols].sort_values('Allocation %', ascending=False),
                    hide_index=True, use_container_width=True, key="res_table"
                )
                
            with c2:
                st.subheader("📊 策略視覺化")
                
                if strategy == "相對價值 (Relative Value)" and not df_with_alpha.empty:
                    base_data = df_with_alpha
                    x_range = np.linspace(base_data['Duration'].min(), base_data['Duration'].max(), 100)
                    try:
                        popt, _ = curve_fit(fit_yield_curve, base_data['Duration'], base_data['YTM'])
                        y_fair = fit_yield_curve(x_range, *popt)
                    except:
                        z = np.polyfit(base_data['Duration'], base_data['YTM'], 2)
                        p = np.poly1d(z)
                        y_fair = p(x_range)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=base_data['Duration'], y=base_data['YTM'],
                        mode='markers', name='市場債券',
                        marker=dict(color='lightgrey', size=8),
                        text=base_data['Name']
                    ))
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_fair,
                        mode='lines', name='合理價值曲線 (Fair Value)',
                        line=dict(color='blue', dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=portfolio['Duration'], y=portfolio['YTM'],
                        mode='markers', name='被低估債券 (Buy)',
                        marker=dict(color='red', size=15, symbol='star'),
                        text=portfolio['Name']
                    ))
                    
                    # 這裡加上一條垂直線，標示使用者的篩選門檻
                    fig.add_vline(x=min_dur, line_width=1, line_dash="dash", line_color="green", annotation_text=f"篩選: >{min_dur}年")
                    
                    fig.update_layout(title="相對價值模型 (尋找曲線上方)", xaxis_title="Duration", yaxis_title="YTM")
                    st.plotly_chart(fig, use_container_width=True, key="rv_chart")
                    
                else:
                    df_raw['Type'] = '未選入'
                    portfolio['Type'] = '建議買入'
                    if excluded_issuers: df_raw.loc[df_raw['Name'].isin(excluded_issuers), 'Type'] = '已剔除'
                    
                    plot_base = df_raw[~df_raw['ISIN'].isin(portfolio['ISIN'])]
                    all_plot = pd.concat([plot_base, portfolio])
                    
                    color_map = {'未選入': '#e0e0e0', '建議買入': '#ef553b', '已剔除': 'rgba(0,0,0,0.1)'}
                    fig = px.scatter(
                        all_plot, x='Duration', y='YTM', color='Type',
                        color_discrete_map=color_map,
                        size=all_plot['Type'].map({'未選入': 5, '建議買入': 15, '已剔除': 3}),
                        hover_data=['Name', 'ISIN'],
                        title=f"目前策略: {strategy}"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="main_chart")
                
        elif uploaded_file and st.session_state.get('last_run'):
            st.warning("⚠️ 找不到符合條件的債券。")

else:
    st.info("👈 請先在左側上傳 Excel 檔案")
