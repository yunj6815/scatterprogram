import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform


# 1. 한글 폰트 설정
def set_korean_font():
    system_name = platform.system()
    if system_name == 'Darwin':
        plt.rc('font', family='AppleGothic')
    elif system_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    else:
        plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False


# 2. 데이터 로드
@st.cache_data
def load_data(file):
    return pd.read_csv(file)


# 3. 통합 시각화 함수 (데이터 타입에 따라 그래프 종류 자동 변경)
def draw_smart_plot(df, x_col, y_col, color_col, show_trendline):
    fig, ax = plt.subplots(figsize=(6, 5))
    hue_val = color_col if color_col != "선택 안함" else None

    is_x_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    is_y_numeric = pd.api.types.is_numeric_dtype(df[y_col])

    # Case 1: 둘 다 숫자형일 때 (표준 산점도)
    if is_x_numeric and is_y_numeric:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_val, ax=ax)
        if show_trendline:
            sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color='red', ax=ax)

    # Case 2: X가 문자형, Y가 숫자형일 때 (범주형 분석)
    elif not is_x_numeric and is_y_numeric:
        # 데이터가 겹치지 않게 옆으로 퍼뜨려 주는 stripplot 사용
        sns.stripplot(data=df, x=x_col, y=y_col, hue=hue_val, jitter=True, alpha=0.7, ax=ax)
        # 박스플롯을 연하게 깔아주면 분포 확인에 용이함
        sns.boxplot(data=df, x=x_col, y=y_col, color='lightgray', width=0.3, ax=ax, fliersize=0)

        if show_trendline:
            st.warning("X축이 문자(범주)인 경우 추세선을 표시할 수 없습니다.")

    # Case 3: 기타 (Y가 문자인 경우 등)
    else:
        st.error("Y축은 반드시 수치 데이터여야 시각화가 가능합니다.")

    return fig


# 4. 스마트 분석 보고서 함수
def display_smart_report(df, x_col, y_col):
    is_x_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    is_y_numeric = pd.api.types.is_numeric_dtype(df[y_col])

    if not is_y_numeric:
        st.write("반응 변수(Y축)가 숫자 데이터가 아니어서 분석을 진행할 수 없습니다.")
        return

    df_clean = df[[x_col, y_col]].dropna()

    # --- 숫자 vs 숫자 보고서 ---
    if is_x_numeric:
        corr = df_clean[x_col].corr(df_clean[y_col])
        st.markdown(f"### 1. 상관관계 분석 ({x_col} - {y_col})")
        direction = "양의" if corr > 0 else "음의"
        strength = "강한" if abs(corr) >= 0.7 else "어느 정도의" if abs(corr) >= 0.3 else "약한"
        st.write(f"두 변수 간 상관계수는 {corr:.2f}로, **{strength} {direction} 상관관계**가 관찰됩니다.")

        st.markdown("### 2. 특이사항 및 결론")
        st.info(f"'{x_col}' 수치가 증가함에 따라 '{y_col}'이(가) 어떻게 변화하는지 경향성이 확인되었습니다.")

    # --- 문자(전공 등) vs 숫자 보고서 ---
    else:
        st.markdown(f"### 1. 그룹별 비교 분석 ({x_col}에 따른 {y_col})")
        group_stats = df_clean.groupby(x_col)[y_col].agg(['mean', 'median', 'count']).sort_values('mean',
                                                                                                  ascending=False)
        st.write(f"각 {x_col} 그룹별로 '{y_col}'의 평균 점수를 산출한 결과입니다.")
        st.dataframe(group_stats.rename(columns={'mean': '평균', 'median': '중앙값', 'count': '인원수'}),
                     use_container_width=True)

        top_group = group_stats.index[0]
        bottom_group = group_stats.index[-1]

        st.markdown("### 2. 분포 요약 및 결론")
        st.write(f"- **최고 성취 그룹:** 평균적으로 '{top_group}' 그룹이 가장 높은 수치를 기록했습니다.")
        st.write(f"- **최저 성취 그룹:** '{bottom_group}' 그룹이 상대적으로 낮은 수치를 보였습니다.")
        st.info(f"그룹별 편차를 확인한 결과, {x_col} 종류가 {y_col} 결과에 영향을 주는 주요 요인임을 알 수 있습니다.")


# 5. 메인 함수
def main():
    st.set_page_config(layout="wide")
    set_korean_font()
    st.title("산점도 및 범주형 데이터 활용하기")

    uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        columns = df.columns.tolist()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**분석 대상 컬럼 선택**")
            x_col = st.selectbox("x축 (설명 변수/전공 등)", columns)
            y_col = st.selectbox("y축 (반응 변수/점수 등)", columns)
            color_options = ["선택 안함"] + columns
            color_col = st.selectbox("색상 구분 범주(선택)", color_options)
            show_trendline = st.checkbox("추세선(회귀선) 표시 (숫자형 데이터 전용)")

        with col2:
            st.subheader(f"'{x_col}'와(과) '{y_col}'의 데이터 분포")
            fig = draw_smart_plot(df, x_col, y_col, color_col, show_trendline)
            st.pyplot(fig, use_container_width=True)

        st.divider()
        st.subheader("데이터 분석 보고서")
        display_smart_report(df, x_col, y_col)


if __name__ == "__main__":
    main()