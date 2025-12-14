import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px

# --- 1. 頁面設定 ---
st.set_page_config(page_title="債券組合優化器 (Bond Optimizer)", layout="wide")

st.title("🛡️ 債券投資組合優化器 (Yield Max Strategy)")
st.markdown("""
此工具使用 **線性規劃 (Linear Programming)**，在滿足「存續期間」與「信用評等」限制下，
尋找能提供 **最大化殖利率 (Max YTM)** 的最佳配置。
""")

# --- 2. 模擬數據生成 (Mock Data) ---
@st.cache_data
def get_bond_data():
    data = {
        'Bond_Name': [
            'US Treasury 10Y', 'US Treasury 2Y', 
            'Apple Corp (AA)', 'Microsoft (AAA)', 'JPM Chase (A)', 
            'Ford Motor (BB)', 'Energy ETF (B)', 'Telekom Bond (BBB)',
            'Short-Term Corp (A)', 'Long-Term Infra (BBB)'
        ],
        'Sector': ['Gov', 'Gov', 'Tech', 'Tech', 'Finance', 'Auto', 'Energy', 'Telecom', 'Finance', 'Utility'],
        'YTM': [0.038, 0.042, 0.051, 0.049, 0.056, 0.078, 0.085, 0.062, 0.053, 0.065],
        'Duration': [8.5, 1.8, 7.2, 9.0, 5.5, 4.2, 5.0, 6.8, 2.5, 12.0],
        'Credit_Score': [1, 1, 2, 1, 3, 5, 6, 4, 3, 4] 
        # Score Logic: 1=AAA/Gov, 2=AA, 3=A, 4=BBB, 5=BB, 6=B
    }
    return pd.DataFrame(data)

df = get_bond_data()

# 信評文字對照表 (用於顯示)
credit_map = {1: 'AAA/Gov', 2: 'AA', 3: 'A', 4: 'BBB', 5: 'BB', 6: 'B'}
df['Credit_Rating'] = df['Credit_Score'].map(credit_map)

# --- 3. 側邊欄：使用者參數設定 ---
st.sidebar.header("⚙️ 優化限制參數")

target_duration = st.sidebar.slider(
    "目標存續期間上限 (Target Duration)", 
    min_value=2.0, max_value=10.0, value=6.0, step=0.5,
    help="投資組合的加權平均存續期間將小於此數值 (控制利率風險)"
)

target_credit = st.sidebar.slider(
    "目標平均信評分數上限", 
    min_value=1.0, max_value=5.0, value=3.5, step=0.1,
    help="1=AAA, 3=A, 4=BBB, 5=BB。數值越低信評越好。"
)
st.sidebar.caption(f"目前設定相當於平均信評約: {credit_map.get(int(round(target_credit)), 'Mix')}")

max_single_weight = st.sidebar.slider(
    "單檔債券持倉上限", 
    min_value=0.1, max_value=1.0, value=0.3, step=0.05,
    help="避免過度集中於單一債券"
)

# --- 4. 優化核心邏輯 (Solver) ---
def optimize_portfolio(df, max_dur, max_credit, max_weight):
    n_bonds = len(df)
    
    # 目標：Maximize YTM => Minimize (-YTM)
    c = -1 * df['YTM'].values
    
    # 不等式限制 (Ax <= b)
    # 1. Duration <= max_dur
    # 2. Credit Score <= max_credit
    A_ub = np.array([
        df['Duration'].values,
        df['Credit_Score'].values
    ])
    b_ub = np.array([max_dur, max_credit])
    
    # 等式限制 (Ax = b): 權重總和 = 1
    A_eq = np.array([np.ones(n_bonds)])
    b_eq = np.array([1.0])
    
    # 邊界: 0 <= weight <= max_weight
    bounds = [(0, max_weight) for _ in range(n_bonds)]
    
    # 求解
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    return result

# 執行按鈕
if st.sidebar.button("🚀 開始計算最佳組合"):
    result = optimize_portfolio(df, target_duration, target_credit, max_single_weight)
    
    if result.success:
        st.success("✅ 優化成功！已找到最佳配置。")
        
        # 處理結果
        df['Optimal_Weight'] = result.x
        portfolio = df[df['Optimal_Weight'] > 0.001].copy()
        portfolio['Allocation %'] = (portfolio['Optimal_Weight'] * 100).round(2)
        
        # 計算組合整體指標
        port_ytm = (portfolio['YTM'] * portfolio['Optimal_Weight']).sum()
        port_dur = (portfolio['Duration'] * portfolio['Optimal_Weight']).sum()
        port_credit = (portfolio['Credit_Score'] * portfolio['Optimal_Weight']).sum()
        
        # --- 5. 顯示結果 ---
        
        # KPI 指標卡
        col1, col2, col3 = st.columns(3)
        col1.metric("預期殖利率 (Yield)", f"{port_ytm:.2%}", delta="最大化目標")
        col2.metric("平均存續期間 (Duration)", f"{port_dur:.2f} 年", delta=f"限制 < {target_duration}")
        col3.metric("平均信評分數", f"{port_credit:.2f}", delta=f"限制 < {target_credit}")
        
        st.divider()

        # 版面配置：左圖右表
        chart_col, table_col = st.columns([1, 1])
        
        with table_col:
            st.subheader("📋 建議持倉明細")
            display_cols = ['Bond_Name', 'Credit_Rating', 'YTM', 'Duration', 'Allocation %']
            
            # 格式化顯示
            st.dataframe(
                portfolio[display_cols].sort_values(by='Allocation %', ascending=False),
                hide_index=True,
                use_container_width=True
            )
            
            # 圓餅圖
            fig_pie = px.pie(portfolio, values='Allocation %', names='Bond_Name', title='資產配置比例')
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col:
            st.subheader("📊 風險/報酬定位圖")
            
            # 建立散佈圖數據：所有債券 + 最佳組合
            plot_df = df.copy()
            plot_df['Type'] = '個別債券'
            plot_df['Size'] = 10
            
            # 新增一行代表「最佳組合」
            new_row = {
                'Bond_Name': '★ 最佳優化組合',
                'YTM': port_ytm,
                'Duration': port_dur,
                'Type': 'Optimized Portfolio',
                'Size': 25,
                'Credit_Rating': 'Mix'
            }
            # 使用 pd.concat 替代 append
            plot_df = pd.concat([plot_df, pd.DataFrame([new_row])], ignore_index=True)

            # 繪圖 (X=Duration/Risk, Y=YTM/Return)
            fig_scatter = px.scatter(
                plot_df, 
                x='Duration', 
                y='YTM', 
                color='Type',
                size='Size',
                hover_data=['Bond_Name', 'Credit_Rating'],
                color_discrete_map={'個別債券': '#636EFA', 'Optimized Portfolio': '#EF553B'},
                title="YTM vs Duration (尋找效率前緣)"
            )
            
            # 加入限制線 (視覺化邊界)
            fig_scatter.add_vline(x=target_duration, line_dash="dash", line_color="green", annotation_text="Duration Limit")
            fig_scatter.update_layout(yaxis_tickformat='.1%')
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.info("💡 說明：紅點是優化後的組合。它通常會位於所有藍點連線的上方邊界（效率前緣），代表在相同的存續期間風險下，獲得了最高的殖利率。")

    else:
        st.error("❌ 無法找到可行解！")
        st.warning("""
        原因可能是限制條件過於嚴格。
        建議嘗試：
        1. 提高「目標存續期間上限」
        2. 提高「目標平均信評分數」（接受較低的信評）
        3. 提高「單檔債券持倉上限」
        """)
        
else:
    st.info("👈 請調整左側參數並點擊按鈕開始計算")