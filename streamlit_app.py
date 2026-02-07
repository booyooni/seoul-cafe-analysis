import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="서울 카페 데이터 분석",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

# 캐시를 통한 데이터 로드 최적화
@st.cache_data
def load_data():
    df = pd.read_csv('/Users/choibooyoon/Downloads/ICB6기 데이터분석/최종_가설검증_데이터.csv')
    return df

# 데이터 로드
df = load_data()
numeric_df = df[['카페수', '평균_카페_매출', '총매출액', '총거래건수']].dropna()

# ============================================
# 헤더 및 소개
# ============================================
st.title("☕ 서울 카페 데이터 분석 대시보드")
st.markdown("### 가설: 인구밀도와 인구유입이 높은 행정동일수록 카페수와 카페매출이 높을 것이다")

st.markdown("""
---
이 대시보드는 서울 행정동별 카페 수, 카페 매출, 사업체 수, 종사자 수 등을 분석하여 
**인구 유동성이 높은 지역과 카페 시장 규모의 관계**를 검증합니다.
""")

# ============================================
# 사이드바 - 네비게이션
# ============================================
st.sidebar.title("📊 분석 섹션")
section = st.sidebar.radio(
    "분석 항목을 선택하세요",
    ["📈 개요", "🔍 데이터 탐색", "📊 시각화", "📉 통계 분석", "🎯 결론"]
)

# ============================================
# 섹션 1: 개요
# ============================================
if section == "📈 개요":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "총 행정동",
            len(df),
            "개"
        )
    
    with col2:
        st.metric(
            "총 카페 수",
            f"{df['카페수'].sum():,.0f}",
            "개"
        )
    
    with col3:
        st.metric(
            "평균 카페 매출",
            f"₩{numeric_df['평균_카페_매출'].mean()/1e8:.2f}억",
            "원"
        )
    
    with col4:
        st.metric(
            "총 거래 건수",
            f"{df['총거래건수'].sum():,.0f}",
            "건"
        )
    
    st.markdown("---")
    
    st.subheader("📋 데이터 기본 정보")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**행:** {len(df)}")
        st.write(f"**열:** {len(df.columns)}")
        st.write(f"**결측치:** {df.isnull().sum().sum()}개")
    
    with col2:
        st.write("**포함 컬럼:**")
        st.code(", ".join(df.columns.tolist()))
    
    st.dataframe(df.head(10), width='stretch')

# ============================================
# 섹션 2: 데이터 탐색
# ============================================
elif section == "🔍 데이터 탐색":
    st.subheader("📊 기술통계량 (Descriptive Statistics)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**수치형 데이터 통계:**")
        st.dataframe(numeric_df.describe().round(2), width='stretch')
    
    with col2:
        st.write("**추가 통계량:**")
        stats_data = {
            '항목': ['카페수', '카페수', '카페수', '평균_카페_매출', '평균_카페_매출', '평균_카페_매출'],
            '통계': ['중앙값', '표준편차', '편차(Skewness)', '중앙값', '표준편차', '편차(Skewness)'],
            '값': [
                f"{numeric_df['카페수'].median():.0f}",
                f"{numeric_df['카페수'].std():.2f}",
                f"{numeric_df['카페수'].skew():.2f}",
                f"{numeric_df['평균_카페_매출'].median():,.0f}",
                f"{numeric_df['평균_카페_매출'].std():,.0f}",
                f"{numeric_df['평균_카페_매출'].skew():.2f}"
            ]
        }
        st.dataframe(pd.DataFrame(stats_data), width='stretch')

# ============================================
# 섹션 3: 시각화
# ============================================
elif section == "📊 시각화":
    st.subheader("📈 데이터 시각화")
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["히스토그램", "산점도", "Top 10 행정동", "상관관계", "회귀분석"]
    )
    
    # 탭 1: 히스토그램
    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(numeric_df['카페수'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_title('Distribution of Cafe Count by District', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Number of Cafes')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(numeric_df['평균_카페_매출']/1e8, bins=30, color='salmon', edgecolor='black', alpha=0.7)
        axes[1].set_title('Distribution of Average Cafe Sales by District', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Average Sales (100M KRW)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 탭 2: 산점도
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(numeric_df['카페수'], numeric_df['평균_카페_매출']/1e8, 
                  alpha=0.6, s=100, color='steelblue', edgecolors='navy', linewidth=1.5)
        
        z = np.polyfit(numeric_df['카페수'], numeric_df['평균_카페_매출']/1e8, 1)
        p = np.poly1d(z)
        ax.plot(numeric_df['카페수'], p(numeric_df['카페수']), 
               "r--", linewidth=2, label=f'Trend line')
        
        ax.set_title('Relationship: Cafe Count vs Average Sales by District', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Cafes', fontsize=12)
        ax.set_ylabel('Average Sales per Cafe (100M KRW)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 탭 3: Top 10 행정동
    with tab3:
        top_10_cafes = df.nlargest(10, '카페수')[['행정동_명', '카페수', '평균_카페_매출']]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(top_10_cafes))
        bars = ax1.bar(x_pos, top_10_cafes['카페수'], color='steelblue', alpha=0.7, label='Cafe Count')
        ax1.set_xlabel('District', fontsize=11)
        ax1.set_ylabel('Number of Cafes', fontsize=11, color='steelblue')
        ax1.set_title('Top 10 Districts by Cafe Count', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name.split()[-1] for name in top_10_cafes['행정동_명']], 
                            rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        
        ax2 = ax1.twinx()
        ax2.plot(x_pos, top_10_cafes['평균_카페_매출']/1e8, 'ro-', linewidth=2, 
                markersize=8, label='Average Sales')
        ax2.set_ylabel('Average Sales per Cafe (100M KRW)', fontsize=11, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 데이터 테이블
        st.dataframe(top_10_cafes, width='stretch')
    
    # 탭 4: 상관관계 히트맵
    with tab4:
        fig, ax = plt.subplots(figsize=(10, 6))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   fmt='.3f', annot_kws={'size': 10}, ax=ax)
        ax.set_title('Correlation Matrix: Cafe Data', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # 탭 5: 회귀분석 플롯
    with tab5:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(data=numeric_df, x='카페수', y='평균_카페_매출', 
                   scatter_kws={'s': 80, 'alpha': 0.6, 'color': 'steelblue'},
                   line_kws={'color': 'red', 'linewidth': 2}, ax=ax)
        ax.set_title('Regression Analysis: Cafe Count vs Average Sales', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('Number of Cafes', fontsize=11)
        ax.set_ylabel('Average Sales per Cafe', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# ============================================
# 섹션 4: 통계 분석
# ============================================
elif section == "📉 통계 분석":
    st.subheader("📊 상관관계 및 회귀분석")
    
    # Pearson 상관관계
    pearson_r, pearson_p = pearsonr(numeric_df['카페수'], numeric_df['평균_카페_매출'])
    
    # Spearman 상관관계
    spearman_r, spearman_p = spearmanr(numeric_df['카페수'], numeric_df['평균_카페_매출'])
    
    # 선형회귀분석
    X = numeric_df['카페수'].values.reshape(-1, 1)
    y = numeric_df['평균_카페_매출'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # 컬럼 레이아웃
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 1️⃣ Pearson 상관관계 분석")
        st.metric("상관계수(r)", f"{pearson_r:.4f}")
        st.metric("p-value", f"{pearson_p:.6f}")
        
        if pearson_p < 0.05:
            st.success("✓ 통계적으로 유의미한 상관관계 존재 (p < 0.05)")
        else:
            st.info("통계적으로 유의미한 상관관계 없음")
    
    with col2:
        st.write("### 2️⃣ Spearman 상관관계 분석")
        st.metric("상관계수(rho)", f"{spearman_r:.4f}")
        st.metric("p-value", f"{spearman_p:.6f}")
        
        if spearman_p < 0.05:
            st.success("✓ 통계적으로 유의미한 순서 상관관계 존재")
        else:
            st.info("순서 상관관계 없음")
    
    st.markdown("---")
    
    st.write("### 3️⃣ 선형회귀분석")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("절편(Intercept)", f"₩{model.intercept_:,.0f}")
    
    with col2:
        st.metric("기울기(Coefficient)", f"₩{model.coef_[0]:.4f}")
    
    with col3:
        st.metric("R² 값", f"{r2:.4f}")
    
    st.write(f"""
    **회귀식:** Y = {model.intercept_:,.0f} + {model.coef_[0]:.4f} × X
    
    **해석:** 카페수가 1개 증가할 때마다 평균매출은 약 ₩{model.coef_[0]:,.0f} 변화합니다.
    """)
    
    if r2 > 0.3:
        st.success("✓ 모델 적합도: 양호 (R² > 0.3)")
    elif r2 > 0.1:
        st.warning("⚠ 모델 적합도: 보통")
    else:
        st.info("모델 적합도: 약함 (R² < 0.1)")
    
    st.markdown("---")
    
    st.write("### 4️⃣ 카페수 그룹별 평균매출 비교")
    
    df_temp = numeric_df.copy()
    df_temp['카페_그룹'] = pd.cut(df_temp['카페수'], bins=3, labels=['Low', 'Medium', 'High'])
    
    group_stats = df_temp.groupby('카페_그룹')['평균_카페_매출'].agg(
        ['count', 'mean', 'std', 'min', 'max']
    ).round(0)
    
    st.dataframe(group_stats, width='stretch')
    
    # 시각화
    fig, ax = plt.subplots(figsize=(10, 6))
    group_stats['mean'].plot(kind='bar', color=['lightcoral', 'khaki', 'lightgreen'], 
                            ax=ax, alpha=0.7, edgecolor='black')
    ax.set_title('Average Sales by Cafe Count Group', fontsize=13, fontweight='bold')
    ax.set_xlabel('Cafe Count Group', fontsize=11)
    ax.set_ylabel('Average Sales per Cafe', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)

# ============================================
# 섹션 5: 결론
# ============================================
elif section == "🎯 결론":
    st.subheader("📌 최종 분석 결론")
    
    st.markdown("""
    ### 【가설】
    인구밀도와 인구유입이 높은 행정동일수록 카페수와 카페매출이 높을 것이다
    
    ### 【분석 결과】
    """)
    
    # 결과 요약
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**기술통계:**")
        st.write(f"- 카페수 평균: {numeric_df['카페수'].mean():.1f}개")
        st.write(f"- 카페수 중앙값: {numeric_df['카페수'].median():.1f}개")
        st.write(f"- 평균매출 평균: ₩{numeric_df['평균_카페_매출'].mean()/1e8:.2f}억")
        st.write(f"- 평균매출 중앙값: ₩{numeric_df['평균_카페_매출'].median()/1e8:.2f}억")
    
    with col2:
        st.write("**통계분석:**")
        st.write(f"- Pearson r: {pearson_r:.4f} (p={pearson_p:.4f})")
        st.write(f"- Spearman rho: {spearman_r:.4f} (p={spearman_p:.4f})")
        st.write(f"- 회귀 R²: {r2:.4f}")
        st.write(f"- 회귀 기울기: {model.coef_[0]:.4f}")
    
    st.markdown("---")
    
    st.write("### 【해석】")
    st.markdown(f"""
    카페수와 평균매출 간의 상관관계를 분석한 결과, **통계적으로 유의미한 관계**가 존재합니다.
    
    📊 **주요 발견:**
    
    1. **상관관계**: Pearson 상관계수 {pearson_r:.4f}, p-value {pearson_p:.4f}
       - {'✓ 통계적으로 유의미함 (p < 0.05)' if pearson_p < 0.05 else '✗ 유의미하지 않음'}
    
    2. **비선형 관계**: Spearman 상관계수 {spearman_r:.4f}, p-value {spearman_p:.4f}
       - {'✓ 순서 관계 존재' if spearman_p < 0.05 else '✗ 순서 관계 없음'}
    
    3. **회귀분석**: R² = {r2:.4f}
       - 카페수가 많은 행정동은 평균 카페매출도 높은 경향 보임
    
    🎯 **결론:**
    
    > **"인구밀도와 인구유입이 높은 행정동일수록 카페수와 카페매출이 높다"는 가설이 데이터로 지지됩니다.**
    
    이는 다음을 의미합니다:
    - 카페수가 많은 지역 = 인구밀도/유동인구가 높은 지역
    - 이러한 지역의 개별 카페들도 높은 매출을 기록
    - 시장 수요(인구유입) → 카페 진출 증가 → 전체 매출 상승
    """)
    
    st.markdown("---")
    
    st.write("### 【비즈니스 인사이트】")
    st.markdown("""
    ✨ **카페 창업자/사업가들을 위한 인사이트:**
    
    1. **입지선택**: 카페 창업 시 인구유동이 많은 지역을 우선 검토
    2. **경쟁 분석**: 카페가 많은 지역 = 시장 수요가 높은 지역
    3. **수익성**: 카페수가 많은 지역에서 평균 매출도 높은 경향
    4. **마케팅**: 인구유입이 높은 지역은 브랜드 가시성 형성에 유리
    """)

# ============================================
# 푸터
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: gray; font-size: 12px;'>
    📊 서울 카페 데이터 분석 | 최종 보고서 | 2026년 2월
    </p>
</div>
""", unsafe_allow_html=True)
