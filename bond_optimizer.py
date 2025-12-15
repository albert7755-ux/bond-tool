import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 基礎設定 ---
st.set_page_config(page_title="債券策略大師 (Bond Strategy Pro)", layout="wide")

st.title("🛡️ 債券投資組合策略大師")
st.markdown("""
針對高資產客戶設計的三大經典策略：
1. **收益最大化 (Max Yield)**：在風險限制下追求最高配息。
2. **債券梯 (Ladder)**：平均佈局不同年期，打造穩定現金流。
3. **槓鈴策略 (Barbell)**：長短債配置，兼顧流動性與資本利得。
""")

# --- 2. 輔助函式：資料清洗與處理 ---
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
            
        # 欄位標準化
        col_mapping = {}
        for col in df.columns:
            if 'ISIN' in col.upper(): col_mapping[col] = 'ISIN'
            elif '發行' in col or '名稱' in col: col_mapping[col] = 'Name'
            elif 'YTM' in col.upper() or 'YIELD' in col.upper(): col_mapping[col] = 'YTM'
            elif '存續' in col or 'DURATION' in col.upper(): col_mapping[col] = 'Duration'
            elif 'S&P' in col.upper(): col_mapping[col] = 'SP_Rating'
            elif 'FITCH' in col.upper(): col_mapping[col] = 'Fitch_Rating'
        
        df = df.rename(columns=col_mapping)
        
        # 檢查必要欄位
        req_cols = ['ISIN', 'Name', 'YTM', 'Duration']
        if not all(c in df.columns for c in req_cols):
            return None, f"缺少必要欄位，偵測到: {list(df.columns)}"

        # 數值清洗
        df['YTM'] = pd.to_numeric(df['YTM'], errors='coerce')
        df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
        df = df.dropna(subset=['YTM', 'Duration'])
        df = df[df['YTM'] > 0] # 排除負利率

        # 信評處理
        if 'SP_Rating' in df.columns: df['Rating_Source'] = df['SP_Rating']
        elif 'Fitch_Rating' in df.columns: df['Rating_Source'] = df['Fitch_Rating']
        else: df['Rating_Source'] = 'BBB'
        
        df['Rating_Source'] = df['Rating_Source'].astype(str).str.strip().str.upper()
        df['Credit_Score'] = df['Rating_Source'].map(rating_map).fillna(10)
        
        return df, None
    except Exception as e:
        return None, str(e)

# --- 3. 策略邏輯 ---

def run_max_yield(df, target_dur, target_score, max_w):
    """策略 A: 線性規劃求最大收益"""
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

def run_ladder(df, steps):
    """策略 B: 債券梯 (分籃子挑選最高息)"""
    selected = []
    # steps 範例: [(0,3), (3,5), (5,7), (7,10)]
    
    weight_per_step = 1.0 / len(steps) # 平均分配權重
    
    for (min_d, max_d) in steps:
        # 在該區間內找 YTM 最高的
        pool = df[(df['Duration'] >= min_d) & (df['Duration'] < max_d)]
        if not pool.empty:
            # 挑選 YTM 最高的 1 檔 (你也可以改成挑前 2 檔平分)
            best_bond = pool.loc[pool['YTM'].idxmax()].copy()
            best_bond['Weight'] = weight_per_step
            selected.append(best_bond)
        else:
            # 該區間無債券，權重就會浪費掉 (實務上可再平均分給其他區間，這裡先簡化)
            pass
            
    if selected:
        return pd.DataFrame(selected)
    return pd.DataFrame()

def run_barbell(df, short_limit, long_limit, long_weight):
    """策略 C: 槓鈴策略 (兩端挑選)"""
    short_pool = df[df['Duration'] <= short_limit]
    long_pool = df[df['Duration'] >= long_limit]
    
    selected = []
    
    # 短端挑選 YTM 最高的 2 檔 (權重平分)
    if not short_pool.empty:
        short_picks = short_pool.nlargest(2, 'YTM').copy()
        # 短端總權重 = (1 - long_weight)
        short_picks['Weight'] = (1 - long_weight) / len(short_picks)
        selected.append(short_picks)
        
    # 長端挑選 YTM 最高的 2 檔
    if not long_pool.empty:
        long_picks = long_pool.nlargest(2, 'YTM').copy()
        # 長端總權重 = long_weight
        long_picks['Weight'] = long_weight / len(long_picks)
        selected.append(long_picks)
    
    if selected:
        return pd.concat(selected)
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
        
        portfolio = pd.DataFrame()
        
        # --- 根據策略顯示不同參數 ---
        if strategy == "收益最大化 (Max Yield)":
            st.sidebar.caption("說明：透過演算法算出最高殖利率組合，適合追求極致收益的客戶。")
            t_dur = st.sidebar.slider("存續期間上限", 2.0, 15.0, 6.0)
            t_cred_label = st.sidebar.select_slider("最低信評要求", options=list(rating_map.keys()), value='BBB')
            t_cred = rating_map[t_cred_label]
            max_w = st.sidebar.slider("單檔上限", 0.05, 0.5, 0.2)
            
            if st.sidebar.button("🚀 計算最佳配置"):
                portfolio = run_max_yield(df_clean, t_dur, t_cred, max_w)

        elif strategy == "債券梯 (Ladder)":
            st.sidebar.caption("說明：資金平均分配在不同年期，每年有資金到期，風險最低。")
            # 預設梯子區間
            ladder_options = {
                "短梯 (1-5年)": [(1,2), (2,3), (3,4), (4,5)],
                "中梯 (3-7年)": [(3,4), (4,5), (5,6), (6,7)],
                "長梯 (5-15年)": [(5,7), (7,10), (10,12), (12,15)]
            }
            ladder_type = st.sidebar.selectbox("選擇梯型結構", list(ladder_options.keys()))
            
            if st.sidebar.button("🚀 建立債券梯"):
                portfolio = run_ladder(df_clean, ladder_options[ladder_type])

        elif strategy == "槓鈴策略 (Barbell)":
            st.sidebar.caption("說明：集中投資極短與極長債，不碰中期債。進可攻退可守。")
            col_s, col_l = st.sidebar.columns(2)
            short_lim = col_s.number_input("短債定義 (年以下)", value=3.0)
            long_lim = col_l.number_input("長債定義 (年以上)", value=10.0)
            
            long_w = st.sidebar.slider("長債資金佔比 (槓鈴偏重)", 0.1, 0.9, 0.5, help="50% 代表長短各半")
            
            if st.sidebar.button("🚀 建立槓鈴組合"):
                portfolio = run_barbell(df_clean, short_lim, long_lim, long_w)

        # --- 5. 結果顯示區 (共用) ---
        if not portfolio.empty:
            portfolio['Allocation %'] = (portfolio['Weight'] * 100).round(1)
            
            # 計算整體數據
            avg_ytm = (portfolio['YTM'] * portfolio['Weight']).sum()
            avg_dur = (portfolio['Duration'] * portfolio['Weight']).sum()
            
            # KPI 看板
            st.divider()
            k1, k2, k3 = st.columns(3)
            k1.metric("預期年化殖利率 (YTM)", f"{avg_ytm:.2f}%")
            k2.metric("平均存續期間", f"{avg_dur:.2f} 年")
            k3.metric("總持倉檔數", f"{len(portfolio)} 檔")
            
            # 左右圖表
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
                st.subheader("📊 策略視覺化 (YTM vs Duration)")
                
                # 繪製散佈圖
                df_clean['Type'] = '未選入'
                portfolio['Type'] = '建議買入'
                all_plot = pd.concat([df_clean, portfolio])
                
                fig = px.scatter(
                    all_plot, x='Duration', y='YTM', color='Type',
                    color_discrete_map={'未選入': '#e0e0e0', '建議買入': '#ef553b'},
                    size=all_plot['Type'].map({'未選入': 5, '建議買入': 15}),
                    hover_data=['Name', 'ISIN'],
                    title=f"目前策略分佈: {strategy}"
                )
                
                # 如果是債券梯或槓鈴，加一些輔助線會更清楚
                if strategy == "槓鈴策略 (Barbell)":
                    fig.add_vrect(x0=0, x1=3.0, fillcolor="green", opacity=0.1, annotation_text="短債區")
                    fig.add_vrect(x0=10.0, x1=20.0, fillcolor="orange", opacity=0.1, annotation_text="長債區")
                
                st.plotly_chart(fig, use_container_width=True, key="main_chart")
                
        elif uploaded_file and st.session_state.get('last_run'): # 簡單防呆
            st.warning("⚠️ 找不到符合條件的債券，請嘗試放寬篩選條件 (例如槓鈴策略的長債定義)。")

else:
    st.info("👈 請先在左側上傳 Excel 檔案")
