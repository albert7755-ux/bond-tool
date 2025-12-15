import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 基礎設定 ---
st.set_page_config(page_title="債券策略大師 Pro (Bond Strategy)", layout="wide")

st.title("🛡️ 債券投資組合策略大師 Pro")
st.markdown("""
針對高資產客戶設計的三大經典策略 (含集中度控管)：
1. **收益最大化 (Max Yield)**：在風險限制下追求最高配息。
2. **債券梯 (Ladder)**：平均佈局不同年期，打造穩定現金流。
3. **槓鈴策略 (Barbell)**：長短債配置，兼顧流動性與資本利得。
""")

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

# --- 3. 策略邏輯核心 (修改過) ---

def run_max_yield(df, target_dur, target_score, max_w):
    """策略 A: 線性規劃 (數學上較難直接加入「名稱不重複」的硬限制，故維持原樣，靠單檔上限控制)"""
    n = len(df)
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
    """策略 B: 債券梯 (加入重複檢查)"""
    selected = []
    used_issuers = set() # 用來記錄已經買過的發行機構
    
    weight_per_step = 1.0 / len(steps)
    
    for (min_d, max_d) in steps:
        # 篩選出該年期的候選池，並按 YTM 從高到低排
        pool = df[(df['Duration'] >= min_d) & (df['Duration'] < max_d)].sort_values('YTM', ascending=False)
        
        found = False
        for idx, row in pool.iterrows():
            # 檢查邏輯：如果允許重複 OR 沒出現過 => 才買入
            if allow_dup or (row['Name'] not in used_issuers):
                best_bond = row.copy()
                best_bond['Weight'] = weight_per_step
                selected.append(best_bond)
                used_issuers.add(row['Name']) # 登記起來
                found = True
                break # 這一階梯買到了，跳出迴圈，去下一個階梯
        
        if not found:
            # 這一層如果都買不到(例如都被買光了)，這裡留空或你可以設計候補邏輯
            pass
            
    if selected:
        return pd.DataFrame(selected)
    return pd.DataFrame()

def run_barbell(df, short_limit, long_limit, long_weight, allow_dup):
    """策略 C: 槓鈴策略 (加入重複檢查)"""
    
    # 先把候選名單依 YTM 排序
    short_pool = df[df['Duration'] <= short_limit].sort_values('YTM', ascending=False)
    long_pool = df[df['Duration'] >= long_limit].sort_values('YTM', ascending=False)
    
    selected = []
    used_issuers = set()
    
    # 1. 挑選短債 (取前 2 名)
    short_picks = []
    for idx, row in short_pool.iterrows():
        if len(short_picks) >= 2: break # 挑滿2檔就停
        if allow_dup or (row['Name'] not in used_issuers):
            row = row.copy()
            # 權重計算：短債總倉位 (1-long_weight) / 2
            row['Weight'] = (1 - long_weight) / 2 
            short_picks.append(row)
            used_issuers.add(row['Name'])
            
    # 2. 挑選長債 (取前 2 名)
    long_picks = []
    for idx, row in long_pool.iterrows():
        if len(long_picks) >= 2: break # 挑滿2檔就停
        if allow_dup or (row['Name'] not in used_issuers):
            row = row.copy()
            # 權重計算：長債總倉位 (long_weight) / 2
            row['Weight'] = long_weight / 2
            long_picks.append(row)
            used_issuers.add(row['Name'])
    
    # 合併結果
    final_list = short_picks + long_picks
    if final_list:
        return pd.DataFrame(final_list)
    return pd.DataFrame()


# --- 4. 主程式 UI ---
st.sidebar.header("📂 步驟 1: 資料匯入")
uploaded_file = st.sidebar.file_uploader("上傳債券清單", type=['xlsx', 'csv'])

if uploaded_file:
    df_clean, err = clean_data(uploaded_file)
    
    if err:
        st.error(f"錯誤: {err}")
    else:
        st.sidebar.success(f"已讀取 {len(df_clean)} 檔債券")
        
        # --- 策略選擇器 ---
        st.sidebar.header("🧠 步驟 2: 選擇策略")
        strategy = st.sidebar.radio(
            "請選擇投資策略：",
            ["收益最大化 (Max Yield)", "債券梯 (Ladder)", "槓鈴策略 (Barbell)"]
        )
        
        # 新增：風控選項 (只在 Ladder 和 Barbell 出現)
        allow_dup = True
        if strategy in ["債券梯 (Ladder)", "槓鈴策略 (Barbell)"]:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🛡️ 集中度風控")
            allow_dup = st.sidebar.checkbox("允許發行機構重複?", value=True, help="若取消勾選，系統會強制挑選不同發行人的債券")
            if not allow_dup:
                st.sidebar.caption("✅ 已啟用：同一機構限購一檔")

        portfolio = pd.DataFrame()
        
        # --- 根據策略執行 ---
        if strategy == "收益最大化 (Max Yield)":
            st.sidebar.caption("說明：透過演算法算出最高殖利率組合。")
            t_dur = st.sidebar.slider("存續期間上限", 2.0, 15.0, 6.0)
            t_cred_label = st.sidebar.select_slider("最低信評要求", options=list(rating_map.keys()), value='BBB')
            t_cred = rating_map[t_cred_label]
            max_w = st.sidebar.slider("單檔上限", 0.05, 0.5, 0.2)
            
            if st.sidebar.button("🚀 計算最佳配置"):
                portfolio = run_max_yield(df_clean, t_dur, t_cred, max_w)

        elif strategy == "債券梯 (Ladder)":
            st.sidebar.caption("說明：資金平均分配在不同年期，每年有資金到期。")
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
            k3.metric("發行機構數", f"{unique_issuers} 家", delta="不重複" if unique_issuers == len(portfolio) else "有重複")
            
            c1, c2 = st.columns([4, 6])
            
            with c1:
                st.subheader("📋 建議清單")
                st.dataframe(
                    portfolio[['Name', 'ISIN', 'Rating_Source', 'YTM', 'Duration', 'Allocation %']]
                    .sort_values('Duration'),
                    hide_index=True,
                    use_container_width=True,
                    key="res_table"
                )
                
            with c2:
                st.subheader("📊 策略視覺化")
                
                df_clean['Type'] = '未選入'
                portfolio['Type'] = '建議買入'
                all_plot = pd.concat([df_clean, portfolio])
                
                fig = px.scatter(
                    all_plot, x='Duration', y='YTM', color='Type',
                    color_discrete_map={'未選入': '#e0e0e0', '建議買入': '#ef553b'},
                    size=all_plot['Type'].map({'未選入': 5, '建議買入': 15}),
                    hover_data=['Name', 'ISIN'],
                    title=f"目前策略: {strategy}"
                )
                
                if strategy == "槓鈴策略 (Barbell)":
                    fig.add_vrect(x0=0, x1=short_lim, fillcolor="green", opacity=0.1, annotation_text="短債")
                    fig.add_vrect(x0=long_lim, x1=20.0, fillcolor="orange", opacity=0.1, annotation_text="長債")
                
                st.plotly_chart(fig, use_container_width=True, key="main_chart")
                
        elif uploaded_file and st.session_state.get('last_run'):
            st.warning("⚠️ 找不到符合條件的債券，請放寬條件 (例如允許重複或調整年期)。")

else:
    st.info("👈 請先在左側上傳 Excel 檔案")
