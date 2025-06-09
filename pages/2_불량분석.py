import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from st_aggrid import AgGrid, GridOptionsBuilder
import math
from collections.abc import Sequence
import plotly.graph_objects as go
import shap
import xgboost as xgb
import streamlit as st
import numpy as np


st.title("과거 데이터 분석 데스크")

# -----------------------------------------
def load_preprocess_model_shap(
    csv_path="data/train.csv",
    drop_cols=None,
    target_col="passorfail",
    n_estimators=100,
    max_depth=4,
    random_state=42,
):
    if drop_cols is None:
        drop_cols = [
            'id', 'date', 'time', 'registration_time',
            'line', 'name', 'mold_name', 'upper_mold_temp3', 'lower_mold_temp3'
        ]
    df = pd.read_csv(csv_path)
    X = df.drop(columns=drop_cols + [target_col], errors='ignore')
    y = df[target_col]
    for col in X.columns:
        mode_val = X[col].mode(dropna=True)
        if not mode_val.empty:
            X[col] = X[col].fillna(mode_val.iloc[0])
        else:
            X[col] = X[col].fillna(0)
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category').cat.codes
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    importances = shap_values.abs.mean(0).values
    feature_names = X.columns
    sorted_idx = importances.argsort()[::-1]
    top_features = feature_names[sorted_idx][:10]
    return {
        "df": df,
        "X": X,
        "y": y,
        "model": model,
        "explainer": explainer,
        "shap_values": shap_values,
        "importances": importances,
        "feature_names": feature_names,
        "top_features": top_features,
    }

@st.cache_resource
def get_shap_all():
    return load_preprocess_model_shap(csv_path="data/train.csv")
shap_results = get_shap_all()

# -----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    df["registration_time"] = pd.to_datetime(df['date'] + " " + df['time'])
    return df

df = load_data()

# 클러스터 할당 (주요 변수로 군집화, k=10 예시)
main_cols = ['cast_pressure', 'molten_temp', 'lower_mold_temp1', 'upper_mold_temp1']
X = df[main_cols].copy()
for col in main_cols:
    mode_val = X[col].mode(dropna=True)
    if not mode_val.empty:
        X[col] = X[col].fillna(mode_val.iloc[0])
    else:
        X[col] = X[col].fillna(0)
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 페이지 전체를 col_main1(좌), col_main2(우)로 분할
col_main1, col_main2 = st.columns([1, 2])


# 상단 필터 2개를 2-컬럼으로 나란히
with col_main1:
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        selected_ids = st.multiselect('ID 필터(복수)', df['id'].unique())
    with col_f2:
        min_time = df['registration_time'].min()
        max_time = df['registration_time'].max()
        date_range = st.date_input(
            '등록일자 범위',
            [min_time.date(), max_time.date()],
            min_value=min_time.date(),
            max_value=max_time.date()
        )

    filtered = df[
        (df['registration_time'] >= pd.to_datetime(date_range[0])) &
        (df['registration_time'] <= pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)) &
        (df['passorfail'] == 1)  # 🔴 불량만 출력
    ]
    if selected_ids:
        filtered = filtered[filtered['id'].isin(selected_ids)]

    show_cols = ['id', 'registration_time', 'mold_code', 'heating_furnace']

    # 한 페이지당 행 수
    page_size = 10
    total_pages = max(1, math.ceil(len(filtered) / page_size))

    display_df = filtered[show_cols].reset_index(drop=True)
    start_idx = 0
    end_idx = page_size
    if total_pages > 1:
        page_num = st.number_input(
            f"페이지 (1~{total_pages})", min_value=1, max_value=total_pages, value=1, step=1
        )
        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size

    # st-aggrid 표(한 번만!)
    gb = GridOptionsBuilder.from_dataframe(display_df.iloc[start_idx:end_idx])
    gb.configure_selection('single', use_checkbox=False)
    grid_options = gb.build()
    grid_response = AgGrid(
        display_df.iloc[start_idx:end_idx],
        gridOptions=grid_options,
        height=200,
        width='100%',
        theme='streamlit',
        allow_unsafe_jscode=True,
        update_mode='SELECTION_CHANGED',
        fit_columns_on_grid_load=True,
    )

    selected_rows = grid_response['selected_rows']

    default_id = 12493

    if isinstance(selected_rows, pd.DataFrame):
        is_selected = not selected_rows.empty
    elif isinstance(selected_rows, Sequence):
        is_selected = len(selected_rows) > 0
    else:
        is_selected = False

    display_ids = display_df['id'].astype(str).tolist()

    if is_selected:
        # 표에서 선택된 것 사용
        if isinstance(selected_rows, pd.DataFrame):
            sel_id = str(selected_rows.iloc[0]['id'])
        else:
            sel_id = str(selected_rows[0]['id'])
    elif str(default_id) in display_ids:
        sel_id = str(default_id)
    else:
        sel_id = display_ids[0] if display_ids else None

    if sel_id is not None:
        origin_row = df[df['id'].astype(str) == sel_id].iloc[0]
    else:
        st.warning("선택 가능한 데이터가 없습니다.")
    

    # -- 타입 안전하게 체크! (list/DataFrame 둘 다 지원) --
    if isinstance(selected_rows, pd.DataFrame):
        is_selected = not selected_rows.empty
    elif isinstance(selected_rows, Sequence):
        is_selected = len(selected_rows) > 0
    else:
        is_selected = False

    if is_selected:
        if isinstance(selected_rows, pd.DataFrame):
            sel_id = str(selected_rows.iloc[0]['id'])
        else:
            sel_id = str(selected_rows[0]['id'])

        origin_row = df[df['id'].astype(str) == sel_id].iloc[0]
        target_cluster = origin_row['cluster']

        is_defect = df['passorfail'] == 1
        cluster_mask = df['cluster'] == target_cluster

        num_similar_defect = ((cluster_mask) & (is_defect)).sum()
        total_defect = is_defect.sum()
        similar_defect_ratio = num_similar_defect / total_defect if total_defect > 0 else 0

        recent_cut = origin_row['registration_time'] - pd.Timedelta(days=7)
        recent_count = df[
            cluster_mask & is_defect & (df['registration_time'] >= recent_cut)
        ].shape[0]

        main_group = origin_row['mold_code']
        recent_time = df[cluster_mask & is_defect]['registration_time'].max()
        total_prod = df.shape[0]
        similar_defect_rate_in_all = num_similar_defect / total_prod if total_prod > 0 else 0

        st.markdown("#### 선택 불량품 상세 정보")
        st.write(f"유사 불량품 비율: {similar_defect_ratio:.2%}")
        st.write(f"최근 7일 내 유사 불량 발생 횟수: {recent_count}")
        st.write(f"해당 그룹(mold_code): {main_group}")
        st.write(f"최근 발생 시각: {recent_time}")
        st.write(f"전체 생산 대비 비율: {similar_defect_rate_in_all:.2%}")


    else:
        st.info("표에서 불량 데이터를 선택하세요.")


from sklearn.tree import DecisionTreeClassifier
import plotly.graph_objects as go

with col_main2:
    if sel_id is not None:
        sel_mold_code = origin_row['mold_code']
        sel_id = origin_row['id']
        df_mold = df[df['mold_code'] == sel_mold_code].sort_values('registration_time')
        sel_row_df = df_mold[df_mold['id'] == sel_id]

        tab1, tab2 = st.tabs(["금형 온도계열", "압력/슬리브/속도"])

        # 1️⃣ 탭1: upper/lower mold temp 3개
        with tab1:
            for feature in ['upper_mold_temp1', 'lower_mold_temp1', 'lower_mold_temp2']:
                feat_mode = df_mold[feature].mode()
                feat_filled = df_mold[feature].fillna(feat_mode.iloc[0] if not feat_mode.empty else 0)
                X = feat_filled.values.reshape(-1, 1)
                y = df_mold['passorfail']
                split_val = None
                if y.nunique() == 2:
                    tree = DecisionTreeClassifier(max_depth=1, random_state=42)
                    tree.fit(X, y)
                    split_val = tree.tree_.threshold[0]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_mold[df_mold['passorfail'] == 0]['registration_time'],
                    y=df_mold[df_mold['passorfail'] == 0][feature],
                    mode='markers', name='양품(0)', marker=dict(color='blue', size=6, symbol='circle')
                ))
                fig.add_trace(go.Scatter(
                    x=df_mold[df_mold['passorfail'] == 1]['registration_time'],
                    y=df_mold[df_mold['passorfail'] == 1][feature],
                    mode='markers', name='불량(1)', marker=dict(color='red', size=8, symbol='circle')
                ))
                if not sel_row_df.empty:
                    fig.add_trace(go.Scatter(
                        x=sel_row_df['registration_time'],
                        y=sel_row_df[feature],
                        mode='markers', name='선택 행',
                        marker=dict(color='green', size=14, symbol='diamond-open'), showlegend=True
                    ))
                if split_val is not None:
                    fig.add_hline(y=split_val, line_dash="dash", line_color="orange",
                                  annotation_text=f"임계값 {split_val:.2f}")
                fig.update_layout(
                    title=f"{feature} 시계열 (mold_code={sel_mold_code})",
                    xaxis_title="registration_time", yaxis_title=feature,
                    height=280, margin=dict(l=30, r=30, t=40, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

        # 2️⃣ 탭2: cast_pressure, sleeve_temperature, low_section_speed
        with tab2:
            for feature in ['cast_pressure', 'sleeve_temperature', 'low_section_speed']:
                feat_mode = df_mold[feature].mode()
                feat_filled = df_mold[feature].fillna(feat_mode.iloc[0] if not feat_mode.empty else 0)
                X = feat_filled.values.reshape(-1, 1)
                y = df_mold['passorfail']
                split_val = None
                if y.nunique() == 2:
                    tree = DecisionTreeClassifier(max_depth=1, random_state=42)
                    tree.fit(X, y)
                    split_val = tree.tree_.threshold[0]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_mold[df_mold['passorfail'] == 0]['registration_time'],
                    y=df_mold[df_mold['passorfail'] == 0][feature],
                    mode='markers', name='양품(0)', marker=dict(color='blue', size=6, symbol='circle')
                ))
                fig.add_trace(go.Scatter(
                    x=df_mold[df_mold['passorfail'] == 1]['registration_time'],
                    y=df_mold[df_mold['passorfail'] == 1][feature],
                    mode='markers', name='불량(1)', marker=dict(color='red', size=8, symbol='star')
                ))
                if not sel_row_df.empty:
                    fig.add_trace(go.Scatter(
                        x=sel_row_df['registration_time'],
                        y=sel_row_df[feature],
                        mode='markers', name='선택 행',
                        marker=dict(color='green', size=14, symbol='diamond-open'), showlegend=True
                    ))
                if split_val is not None:
                    fig.add_hline(y=split_val, line_dash="dash", line_color="orange",
                                  annotation_text=f"임계값 {split_val:.2f}")
                fig.update_layout(
                    title=f"{feature} 시계열 (mold_code={sel_mold_code})",
                    xaxis_title="registration_time", yaxis_title=feature,
                    height=280, margin=dict(l=30, r=30, t=40, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("좌측 표에서 불량 데이터를 선택하면 시계열 그래프가 표시됩니다.")


#하단
col_bottom1, col_bottom2 = st.columns([1, 2 ])

with col_bottom1:
    shap_df = shap_results["df"]
    X = shap_results["X"]
    shap_values = shap_results["shap_values"]

    # 좌측 표에서 선택한 id 연동
    if 'sel_id' in locals():
        idx_list = shap_df.index[shap_df['id'].astype(str) == str(sel_id)].tolist()
        if len(idx_list) > 0:
            row_idx = idx_list[0]
        else:
            row_idx = 0
    else:
        row_idx = 0

    n = 7  # 상위 몇 개 변수까지 표시
    shap_row = shap_values[row_idx]
    if hasattr(shap_row, "values"):
        shap_row = shap_row.values

    contrib = pd.DataFrame({
        "Feature": X.columns,
        "SHAP_value": shap_row
    }).sort_values("SHAP_value", key=lambda x: np.abs(x), ascending=False)
    top_contrib = contrib.iloc[:n]
    if len(contrib) > n:
        etc_sum = contrib.iloc[n:]['SHAP_value'].sum()
        etc = pd.DataFrame({"Feature": ["기타"], "SHAP_value": [etc_sum]})
        contrib_show = pd.concat([top_contrib, etc], ignore_index=True)
    else:
        contrib_show = top_contrib

    # bar차트/표 모두 중요도 높은 순서대로(위→아래)
    contrib_show = contrib_show[::-1].reset_index(drop=True)
    colors = ['red' if v > 0 else 'blue' for v in contrib_show['SHAP_value']]
    max_abs = contrib_show['SHAP_value'].abs().max()

    # 1) Bar chart (위)
    fig = go.Figure(go.Bar(
        x=contrib_show["SHAP_value"],
        y=contrib_show["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in contrib_show["SHAP_value"]],
        textposition="auto",
        textfont=dict(
            color='black',
            size=13,
            family="Arial Black"
        ),
        insidetextanchor='middle',
    ))
    fig.add_vline(x=0, line_color="gray", line_width=2)
    fig.update_traces(
        marker_line_color='black',
        marker_line_width=2
    )
    fig.update_layout(
        title="SHAP 변수별 기여도",
        xaxis_title="SHAP 값 (불량↑: +, 정상↑: -)",
        yaxis_title=None,
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#f5f5f5",
        font=dict(size=13),
    )

    # x축 제한(바 길이)
    fig.update_xaxes(range=[-max_abs * 1.1, max_abs * 1.1])

    st.plotly_chart(fig, use_container_width=True)

    #  2) SHAP 상위 변수 표 (아래)
    st.markdown("#### SHAP 상위 변수")
    st.dataframe(
        contrib_show[::-1].reset_index(drop=True).style.format({"SHAP_value": "{:+.4f}"}),
        use_container_width=True,
        hide_index=True
    )

with col_bottom2:
    #상위변수 10개
    main_feats = list(shap_results["feature_names"][shap_results["importances"].argsort()[::-1]][:10])
    all_feats = main_feats.copy() 
    # 주요 변수(main_feats)와 전체 변수(all_feats)간 상관관계 DataFrame 생성
    cross_corr = pd.DataFrame(index=main_feats, columns=all_feats)
    for main_feat in main_feats:
        for feat in all_feats:
            vals = X[[main_feat, feat]].dropna()
            # 안전하게 Series만 사용
            v1 = vals[main_feat]
            v2 = vals[feat]
            if isinstance(v1, pd.DataFrame):
                v1 = v1.iloc[:, 0]
            if isinstance(v2, pd.DataFrame):
                v2 = v2.iloc[:, 0]
            if len(v1) > 1 and len(v2) > 1:
                corr_val = v1.corr(v2)
            else:
                corr_val = np.nan
            cross_corr.loc[main_feat, feat] = corr_val
    cross_corr = cross_corr.astype(float)

    # 자기자신(대각선) 0으로 처리
    cross_corr_no_diag = cross_corr.copy()
    for i in range(len(main_feats)):
        if main_feats[i] in all_feats:
            cross_corr_no_diag.iloc[i, all_feats.index(main_feats[i])] = 0

    # 3. 상위 3개 추출
    flat_corr = [
        (main_feats[i], main_feats[j], cross_corr_no_diag.values[i, j])
        for i in range(len(main_feats))
        for j in range(len(main_feats))
        if i < j
    ]
    top3 = [x for x in sorted(flat_corr, key=lambda x: abs(x[2]), reverse=True) if not np.isnan(x[2])][:3]

    # ------ 체크박스 먼저 ------
    show_top3 = st.checkbox("상위 3 상관관계 변수쌍 보기", value=False)
    placeholder = st.empty()
    if show_top3:
        items = [
            f"<span style='font-size:18px'>• <b>{m}</b> ↔ <b>{a}</b> : {v:+.2f}</span>"
            for m, a, v in top3
        ]
        placeholder.markdown("<br>".join(items), unsafe_allow_html=True)
    else:
        # show_top3가 꺼져 있어도 일정한 높이 유지
        placeholder.markdown("<br>" * len(top3), unsafe_allow_html=True)

    # 히트맵
    fig_corr = go.Figure(data=go.Heatmap(
        z=cross_corr.values,
        x=main_feats,
        y=main_feats,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(cross_corr.values, 2),   # 히트맵에 수치만 보여줌
        texttemplate="%{text}",
        colorbar=dict(title="상관계수", tickfont=dict(size=12))
    ))

    #  상위 3개 상관관계쌍에 테두리 강조 추가
    for m, a, v in top3:
        y_idx = main_feats.index(m)
        x_idx = main_feats.index(a)
        fig_corr.add_shape(
            type="rect",
            x0=x_idx - 0.5, x1=x_idx + 0.5,
            y0=y_idx - 0.5, y1=y_idx + 0.5,
            line=dict(color="orange", width=4),
            fillcolor="rgba(0,0,0,0)",
            layer="above"
        )

    fig_corr.update_layout(
        height=360 + len(main_feats)*12,
        margin=dict(l=10, r=10, t=40, b=10),
        title="주요 변수 상관관계 히트맵",
        xaxis_title="전체 변수",
        yaxis_title="주요 변수",
        font=dict(size=13),
        xaxis_tickangle=45,
    )
    st.plotly_chart(fig_corr, use_container_width=True)



    # ------------------------------------------------


    main_features = list(shap_results["feature_names"][shap_results["importances"].argsort()[::-1]][:5])

    # 한 번만 계산해서 dict에 저장
    split_dict = {}
    for feature in main_features:
        feat_mode = df[feature].mode()
        X_feat = df[feature].fillna(feat_mode.iloc[0] if not feat_mode.empty else 0).values.reshape(-1, 1)
        y_feat = df['passorfail']
        split_val = None
        if len(np.unique(y_feat)) == 2:
            tree = DecisionTreeClassifier(max_depth=1, random_state=42)
            tree.fit(X_feat, y_feat)
            split_val = tree.tree_.threshold[0]
        split_dict[feature] = split_val

        











    #  4) 예제/임계값/설명 영역 (하단)
    st.markdown("---")
    st.markdown("#### [예제] 주요 변수 임계값 시각화 영역")
    st.info("여기에 '임계값 시각화' 등 추가 시각화 및 설명을 배치할 수 있습니다.")
