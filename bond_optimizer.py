import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px

# --- 1. 基礎設定 ---
st.set_page_config(page_title="債券組合優化器 Pro (Excel版)", layout="wide")

st.title("🛡️ 債券投資組合優化器 Pro")
st.markdown("""
### 使用說明
請上傳包含以下欄位的 Excel 或 CSV 檔案：
- **ISIN** (或代碼)
- **發行人/保證人** (或名稱)
- **YTM** (殖利率)
- **存續期間** (Duration)
- **S&P** 或 **Fitch** (信用評等)
""")

# --- 2. 輔助函式：信評轉分數 ---
# 我們將 AAA 定義為 1 分，分數越低越好。BBB- 為 10 分。
rating_map = {
    'AAA': 1, 'AA+': 2, 'AA': 3, 'AA-': 4,
    'A+': 5, 'A': 6, 'A-': 7,
    'BBB+': 8, 'BBB': 9, 'BBB-': 10,
    'BB+': 11, 'BB': 12, 'BB-': 13,
    'B+': 14, 'B': 15, 'B-': 16
}

def clean_data(uploaded_file):
    """讀取並清洗使用者上傳的檔案"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"檔案讀取失敗: {e}")
        return None

    # 1. 欄位名稱標準化 (避免欄位名稱有些微差異)
    # 這裡做一個簡單的映射，確保程式能找到對應的欄位
    col_mapping = {}
    for col in df.columns:
        if 'ISIN' in col.upper(): col_mapping[col] = 'ISIN'
        elif '發行' in col or '名稱' in col: col_mapping[col] = 'Name'
        elif 'YTM' in col.upper() or 'YIELD' in col.upper(): col_mapping[col] = 'YTM'
        elif '存續' in col or 'DURATION' in col.upper(): col_mapping[col] = 'Duration'
        elif 'S&P' in col.upper(): col_mapping[col] = 'SP_Rating'
        elif 'FITCH' in col.upper(): col_mapping[col] = 'Fitch_Rating'
    
    df = df.rename(columns=col_mapping)
    
    # 檢查必要欄位是否存在
    required_cols = ['ISIN', 'Name', 'YTM', 'Duration']
    if not all(col in df.columns for col in required_cols):
        st.error(f"錯誤：檔案缺少必要欄位。偵測到的欄位：{list(df.columns)}")
        return None

    # 2. 數據清洗
    # 強制將數值欄位轉為數字，無法轉的 (如文字) 變 NaN
    df['YTM'] = pd.to_numeric(df['YTM'], errors='coerce')
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
    
    # 移除 YTM 或 Duration 是空值的行 (這會自動過濾掉檔案中間的髒文字)
    df = df.dropna(subset=['YTM', 'Duration'])
    
    # 移除 YTM <= 0 的行 (負利率或錯誤數據)
    df = df[df['YTM'] > 0]

    # 3. 處理信評 (文字轉數字)
    # 優先使用 S&P，如果沒有則用 Fitch
    if 'SP_Rating' in df.columns:
        df['Rating_Source'] = df['SP_Rating']
    elif 'Fitch_Rating' in df.columns:
        df['Rating_Source'] = df['Fitch_Rating']
    else:
        # 如果都沒有信評，預設給 BBB (9分) 以免程式崩潰，但在實務上應剔除
        df['Rating_Source'] = 'BBB' 

    # 將文字信評去除空白並轉大寫
    df['Rating_Source'] = df['Rating_Source'].astype(str).str.strip().str.upper()
    
    # 映射為分數
    df['Credit_Score'] = df['Rating_Source'].map(rating_map)
    
    # 如果對應不到 (例如沒信評)，填入 10 (BBB-) 或是選擇剔除
    df['Credit_Score'] = df['Credit_Score'].fillna(10)

    # 顯示給使用者看用的信評 (反向查找)
    # 為了方便，我們直接保留原始文字
    
    return df

# --- 3. 側邊欄與檔案上傳 ---
uploaded_file = st.sidebar.file_uploader("📂 步驟 1: 上傳債券清單 (Excel/CSV)", type=['xlsx', 'xls', 'csv'])

if uploaded_file is not None:
    df_clean = clean_data(uploaded_file)
    
    if df_clean is not None:
        st.sidebar.success(f"成功讀取 {len(df_clean)} 檔有效債券！")
        
        # --- 設定參數 ---
        st.sidebar.header("⚙️ 步驟 2: 設定優化目標")
        
        target_duration = st.sidebar.slider("目標存續期間上限 (年)", 2.0, 15.0, 6.0, 0.5)
        
        # 讓使用者選擇信評上限 (顯示文字，但背後傳數字)
        rating_options = list(rating_map.keys())
        target_credit_label = st.sidebar.select_slider(
            "目標平均信評 (最差允許到)", 
            options=rating_options, 
            value='A-' # 預設 A-
        )
        target_credit_score = rating_map[target_credit_label]
        
        max_single_weight = st.sidebar.slider("單檔持倉上限 (%)", 5, 50, 20, 5) / 100.0

        # --- 4. 優化引擎 ---
        if st.sidebar.button("🚀 開始計算最佳組合"):
            # 準備數據
            n_bonds = len(df_clean)
            c = -1 * df_clean['YTM'].values # 目標: Max YTM
            
            # 限制條件
            A_ub = np.array([
                df_clean['Duration'].values,
                df_clean['Credit_Score'].values
            ])
            b_ub = np.array([target_duration, target_credit_score])
            
            A_eq = np.array([np.ones(n_bonds)])
            b_eq = np.array([1.0])
            
            bounds = [(0, max_single_weight) for _ in range(n_bonds)]
            
            # 求解
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.success:
                st.success("✅ 優化成功！")
                
                # 整理結果
                df_clean['Weight'] = res.x
                portfolio = df_clean[df_clean['Weight'] > 0.001].copy()
                portfolio['Allocation %'] = (portfolio['Weight'] * 100).round(2)
                
                # 計算組合數據
                port_ytm = (portfolio['YTM'] * portfolio['Weight']).sum()
                port_dur = (portfolio['Duration'] * portfolio['Weight']).sum()
                
                # 顯示指標
                col1, col2, col3 = st.columns(3)
                col1.metric("預期年化報酬 (YTM)", f"{port_ytm:.2f}%")
                col2.metric("平均存續期間", f"{port_dur:.2f} 年")
                col3.metric("平均信評限制", target_credit_label)
                
                st.divider()
                
                # 左右佈局
                c1, c2 = st.columns([1, 1])
                
                with c1:
                    st.subheader("📋 建議配置清單")
                    st.dataframe(
                        portfolio[['Name', 'ISIN', 'Rating_Source', 'YTM', 'Duration', 'Allocation %']]
                        .sort_values('Allocation %', ascending=False),
                        hide_index=True
                    )
                    
                    # 下載按鈕
                    csv = portfolio.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("下載配置結果 (CSV)", csv, "optimized_portfolio.csv", "text/csv")

                with c2:
                    st.subheader("📊 配置視覺化")
                    fig = px.pie(portfolio, values='Allocation %', names='Name', title='發行人分散比例')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 風險分布圖
                    df_clean['Type'] = '未選入'
                    portfolio['Type'] = '建議買入'
                    plot_data = pd.concat([df_clean, portfolio])
                    
                    fig2 = px.scatter(
                        plot_data, x='Duration', y='YTM', color='Type',
                        color_discrete_map={'未選入': 'lightgrey', '建議買入': 'red'},
                        hover_data=['Name', 'ISIN'],
                        title="市場機會地圖 (YTM vs Duration)"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
            else:
                st.error("❌ 找不到可行解！請嘗試放寬「信評」或「存續期間」的限制。")
    
else:
    st.info("👋 請在左側上傳你的 Excel 或 CSV 檔案以開始分析。")
