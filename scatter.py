# 4. 스마트 상세 분석 보고서 함수 (4단계 심층 분석 복원)
def display_smart_report(df, x_col, y_col):
    is_x_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    is_y_numeric = pd.api.types.is_numeric_dtype(df[y_col])

    if not is_y_numeric:
        st.write("반응 변수(Y축)가 숫자 데이터가 아니어서 통계 분석을 진행할 수 없습니다.")
        return

    # 결측치 제거
    df_clean = df[[x_col, y_col]].dropna().copy()

    # ==========================================
    # [Case 1] 숫자 vs 숫자 상세 보고서 (기존 4단계 복원)
    # ==========================================
    if is_x_numeric:
        corr = df_clean[x_col].corr(df_clean[y_col])

        # --- 1. 상관관계 ---
        st.markdown(f"### 1. 상관관계 분석 ('{x_col}' - '{y_col}')")

        if corr > 0:
            direction, trend = "양의", "증가할수록 대체로 높아지는"
        elif corr < 0:
            direction, trend = "음의", "증가할수록 대체로 낮아지는"
        else:
            direction, trend = "무상관", "뚜렷한 방향성을 찾기 어려운"

        strength = "강한" if abs(corr) >= 0.7 else "어느 정도의" if abs(corr) >= 0.3 else "매우 약한"

        st.write(f"- **전체 흐름:** '{x_col}' 수치가 {trend} 경향이 관찰됩니다.")
        st.write(f"- **상관 강도:** 두 변수의 상관계수는 {corr:.2f}로, **{strength} {direction} 상관관계**를 보입니다.")

        # --- 2. 데이터 분포 요약 ---
        st.markdown("### 2. 데이터 분포 요약")
        st.write(f"'{x_col}'의 크기를 기준으로 전체 데이터를 3개의 구간(하/중/상)으로 나누어 분포를 확인합니다.")
        try:
            df_clean['구간'] = pd.qcut(df_clean[x_col], q=3, labels=['하위 그룹', '중간 그룹', '상위 그룹'], duplicates='drop')
            summary = df_clean.groupby('구간', observed=False)[y_col].agg(['min', 'max', 'mean']).round(1)
            summary.columns = [f'{y_col} 최소값', f'{y_col} 최대값', f'{y_col} 평균값']
            st.dataframe(summary, use_container_width=True)
        except Exception:
            st.write("데이터가 특정 값에 편향되어 구간별 분할 요약을 제공하기 어렵습니다.")

        # --- 3. 특이사항 분석 ---
        st.markdown("### 3. 특이사항 분석")
        q1 = df_clean[y_col].quantile(0.25)
        q3 = df_clean[y_col].quantile(0.75)
        iqr = q3 - q1
        outliers = df_clean[(df_clean[y_col] < (q1 - 1.5 * iqr)) | (df_clean[y_col] > (q3 + 1.5 * iqr))]
        outliers_cnt = len(outliers)

        st.write(f"- **이상치(Outlier) 가능성:** 전체적인 흐름에서 벗어난 통계적 예외 데이터가 **{outliers_cnt}건** 발견되었습니다.")

        # --- 4. 결론 및 요약 ---
        st.markdown("### 4. 결론 및 요약")
        if abs(corr) >= 0.3:
            conclusion = f"'{x_col}'(은)는 '{y_col}'에 영향을 미치는 유의미한 요인입니다. 단, {outliers_cnt}개의 예외 사례를 고려할 때 다른 외부 요인도 존재할 수 있습니다."
        else:
            conclusion = f"현재 데이터로는 '{x_col}'와(과) '{y_col}' 사이의 뚜렷한 관련성을 찾기 어렵습니다. 다른 변수를 탐색해 보는 것을 권장합니다."
        st.info(conclusion)

    # ==========================================
    # [Case 2] 범주(문자) vs 숫자 상세 보고서 (새로운 4단계 적용)
    # ==========================================
    else:
        # --- 1. 그룹별 비교 분석 ---
        st.markdown(f"### 1. 그룹 간 차이 분석 ('{x_col}'에 따른 '{y_col}')")

        # 그룹별 통계 계산
        group_stats = df_clean.groupby(x_col)[y_col].agg(['mean', 'median', 'min', 'max', 'count']).round(1)
        group_stats = group_stats.sort_values('mean', ascending=False)  # 평균 기준 내림차순 정렬

        top_group = group_stats.index[0]
        bottom_group = group_stats.index[-1]
        mean_diff = group_stats['mean'].iloc[0] - group_stats['mean'].iloc[-1]

        st.write(f"- **최고 성취 그룹:** '{top_group}' 그룹이 가장 높은 평균 수치를 기록했습니다.")
        st.write(f"- **최저 성취 그룹:** '{bottom_group}' 그룹이 상대적으로 가장 낮은 평균 수치를 보였습니다.")
        st.write(f"- **최대 격차:** 두 그룹 간의 평균 차이는 **{mean_diff:.1f}** 입니다.")

        # --- 2. 데이터 분포 요약 ---
        st.markdown("### 2. 그룹별 상세 통계 요약")
        st.write(f"각 '{x_col}' 소속 그룹 내에서의 점수 범위와 평균, 중앙값을 확인합니다.")
        st.dataframe(
            group_stats.rename(columns={'mean': '평균', 'median': '중앙값', 'min': '최소값', 'max': '최대값', 'count': '데이터 수'}),
            use_container_width=True)

        # --- 3. 특이사항 분석 ---
        st.markdown("### 3. 특이사항 및 그룹 내 편차 분석")

        # 각 그룹별 이상치 계산
        total_outliers = 0
        outlier_text = []
        for group, data in df_clean.groupby(x_col):
            q1 = data[y_col].quantile(0.25)
            q3 = data[y_col].quantile(0.75)
            iqr = q3 - q1
            outliers = data[(data[y_col] < (q1 - 1.5 * iqr)) | (data[y_col] > (q3 + 1.5 * iqr))]
            if len(outliers) > 0:k
                total_outliers += len(outliers)
                outlier_text.append(f"'{group}' 그룹에서 {len(outliers)}건")

        if total_outliers > 0:
            st.write(
                f"- **이상치(Outlier):** 각 그룹의 평균적인 분포에서 크게 벗어난 예외 데이터가 총 **{total_outliers}건** 발견되었습니다. ({', '.join(outlier_text)})")
            st.write("- **해석:** 동일한 그룹(조건) 내에서도 점수 편차가 매우 큰 샘플이 존재함을 의미합니다. 이는 해당 그룹 내에 성적을 가르는 또 다른 중요 변수가 있음을 시사합니다.")
        else:
            st.write("- **이상치(Outlier):** 각 그룹 내에서 통계적으로 크게 벗어난 극단적인 데이터는 발견되지 않았습니다. 대부분이 그룹 평균 주변에 고르게 분포하고 있습니다.")

        # --- 4. 결론 및 요약 ---
        st.markdown("### 4. 결론 및 요약")
        if mean_diff > (df_clean[y_col].max() - df_clean[y_col].min()) * 0.1:  # 대략적인 유의미한 차이 기준
            conclusion = f"분석 결과, **'{x_col}'의 종류에 따라 '{y_col}' 결과에 유의미한 차이**가 발생하고 있습니다. 이는 '{x_col}'이(가) 결과에 영향을 주는 주요 변수임을 나타냅니다."
        else:
            conclusion = f"분석 결과, **'{x_col}' 그룹 간 '{y_col}' 수치의 차이가 미미**합니다. 해당 변수는 결과에 큰 영향을 주지 않는 것으로 보입니다."

        st.info(conclusion)