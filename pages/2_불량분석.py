import streamlit as st
import pandas as pd
import numpy as np
import math
from collections.abc import Sequence
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import shap
import xgboost as xgb
from st_aggrid import AgGrid, GridOptionsBuilder

# ────────────── 데이터 및 전처리 ──────────────
@st.cache_data
def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    df["registration_time"] = pd.to_datetime(df['date'] + " " + df['time'])
    return df

@st.cache_data
def preprocess_cluster(df, main_cols=None, n_clusters=10):
    if main_cols is None:
        main_cols = [
            'cast_pressure', 'molten_temp', 'lower_mold_temp1',
            'lower_mold_temp2', 'upper_mold_temp1', 'low_section_speed', 'sleeve_temperature'
        ]
    X = df[main_cols].copy()
    for col in X.columns:
        mode_val = X[col].mode(dropna=True)
        X[col] = X[col].fillna(mode_val.iloc[0] if not mode_val.empty else 0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    return df

@st.cache_resource
def get_shap_all(csv_path="data/train.csv"):
    drop_cols = [
        'id', 'date', 'time', 'registration_time',
        'line', 'name', 'mold_name', 'upper_mold_temp3', 'lower_mold_temp3'
    ]
    df = pd.read_csv(csv_path)
    X = df.drop(columns=drop_cols + ['passorfail'], errors='ignore')
    y = df['passorfail']
    for col in X.columns:
        mode_val = X[col].mode(dropna=True)
        X[col] = X[col].fillna(mode_val.iloc[0] if not mode_val.empty else 0)
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category').cat.codes
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    importances = shap_values.abs.mean(0).values
    feature_names = X.columns
    sorted_idx = importances.argsort()[::-1]
    top_features = feature_names[sorted_idx][:10]
    return dict(df=df, X=X, y=y, model=model, explainer=explainer, shap_values=shap_values,
                importances=importances, feature_names=feature_names, top_features=top_features)

# ────────────── 유틸 함수 ──────────────
def get_sel_id(selected_rows, default_id, display_ids):
    if isinstance(selected_rows, pd.DataFrame):
        is_selected = not selected_rows.empty
    elif isinstance(selected_rows, Sequence):
        is_selected = len(selected_rows) > 0
    else:
        is_selected = False
    if is_selected:
        if isinstance(selected_rows, pd.DataFrame):
            return str(selected_rows.iloc[0]['id'])
        else:
            return str(selected_rows[0]['id'])
    elif str(default_id) in display_ids:
        return str(default_id)
    else:
        return display_ids[0] if display_ids else None

def kde_split_threshold(x0, x1):
    """양품/불량값 배열 2개를 받아 KDE로 교차점 임계값 1개 반환 (없으면 None)"""
    if len(x0) < 10 or len(x1) < 10 or np.std(x0) < 1e-6 or np.std(x1) < 1e-6:
        return None
    kde0, kde1 = gaussian_kde(x0), gaussian_kde(x1)
    xx = np.linspace(min(np.min(x0), np.min(x1)), max(np.max(x0), np.max(x1)), 300)
    diff = kde0(xx) - kde1(xx)
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    return (xx[sign_change[0]] + xx[sign_change[0]+1]) / 2 if len(sign_change) > 0 else None

def plot_boxplot_with_thres(vals_0, vals_1, feature, thres=None, title=None):
    fig = go.Figure()
    fig.add_trace(go.Box(y=vals_0, name='양품(0)', marker_color='blue', boxmean=True))
    fig.add_trace(go.Box(y=vals_1, name='불량(1)', marker_color='red', boxmean=True))
    if thres is not None:
        fig.add_hline(y=thres, line_color="orange", line_dash="dash", annotation_text=f"임계값 {thres:.2f}")
    fig.update_layout(title=title or f"{feature} 박스플랏", yaxis_title=feature, height=300)
    return fig

def plot_timeseries_mold(df_mold, sel_row_df, feature, split_val=None, sel_color="lime"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_mold[df_mold['passorfail'] == 0]['registration_time'],
        y=df_mold[df_mold['passorfail'] == 0][feature],
        mode='markers', name='양품(0)', marker=dict(color='blue', size=6, symbol='circle', opacity=0.2)
    ))
    fig.add_trace(go.Scatter(
        x=df_mold[df_mold['passorfail'] == 1]['registration_time'],
        y=df_mold[df_mold['passorfail'] == 1][feature],
        mode='markers', name='불량(1)', marker=dict(color='red', size=8, symbol='circle', opacity=0.2)
    ))
    if not sel_row_df.empty:
        fig.add_trace(go.Scatter(
            x=sel_row_df['registration_time'], y=sel_row_df[feature],
            mode='markers', name='선택 행',
            marker=dict(color=sel_color, size=14, symbol='star', opacity=1), showlegend=True
        ))
    if split_val is not None:
        fig.add_hline(y=split_val, line_dash="dash", line_color="orange", annotation_text=f"임계값 {split_val:.2f}")
    fig.update_layout(title=f"{feature} 시계열", xaxis_title="registration_time", yaxis_title=feature,
                      height=280, margin=dict(l=30, r=30, t=40, b=30),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# ────────────── 메인 함수 (대시보드) ──────────────
def main():
    st.title("불량 데이터 분석 대시보드")
    # --- 데이터 준비
    df = preprocess_cluster(load_data())
    shap_results = get_shap_all()
    # --- 상단(좌): 필터+표+선택ID
    col_main1, col_main2 = st.columns([1, 2])
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
                min_value=min_time.date(), max_value=max_time.date()
            )
        filtered = df[
            (df['registration_time'] >= pd.to_datetime(date_range[0])) &
            (df['registration_time'] <= pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)) &
            (df['passorfail'] == 1)
        ]
        if selected_ids:
            filtered = filtered[filtered['id'].isin(selected_ids)]
        show_cols = ['id', 'registration_time', 'mold_code', 'heating_furnace']
        page_size = 10
        total_pages = max(1, math.ceil(len(filtered) / page_size))
        display_df = filtered[show_cols].reset_index(drop=True)
        start_idx, end_idx = 0, page_size
        if total_pages > 1:
            page_num = st.number_input(
                f"페이지 (1~{total_pages})", min_value=1, max_value=total_pages, value=1, step=1
            )
            start_idx = (page_num - 1) * page_size
            end_idx = start_idx + page_size
        gb = GridOptionsBuilder.from_dataframe(display_df.iloc[start_idx:end_idx])
        gb.configure_selection('single', use_checkbox=False)
        grid_response = AgGrid(
            display_df.iloc[start_idx:end_idx],
            gridOptions=gb.build(),
            height=200, width='100%',
            theme='streamlit', allow_unsafe_jscode=True,
            update_mode='SELECTION_CHANGED', fit_columns_on_grid_load=True,
        )
        selected_rows = grid_response['selected_rows']
        default_id = 12493
        display_ids = display_df['id'].astype(str).tolist()
        sel_id = get_sel_id(selected_rows, default_id, display_ids)
        if sel_id is not None:
            origin_row = df[df['id'].astype(str) == sel_id].iloc[0]
        else:
            st.warning("선택 가능한 데이터가 없습니다.")
            return
        # 정보표시
        if sel_id is not None:
            target_cluster = origin_row['cluster']
            is_defect = df['passorfail'] == 1
            is_same_mold = (df['mold_code'] == origin_row['mold_code'])
            is_same_cluster = (df['cluster'] == origin_row['cluster'])
            num_similar_defect = (is_same_mold & is_same_cluster & is_defect).sum()
            total_defect = (is_same_mold & is_defect).sum()
            similar_defect_ratio = num_similar_defect / total_defect if total_defect > 0 else 0
            recent_cut = origin_row['registration_time'] - pd.Timedelta(days=7)
            is_recent = (df['registration_time'] >= recent_cut)
            recent_similar_defect_count = (is_same_mold & is_same_cluster & is_defect & is_recent).sum()
            main_group = origin_row['mold_code']
            recent_time = df[(is_same_cluster & is_defect)]['registration_time'].max()
            total_prod = df.shape[0]
            similar_defect_rate_in_all = num_similar_defect / total_prod if total_prod > 0 else 0
            st.markdown("#### 선택 불량품 상세 정보")
            st.write(f"유사 불량품 비율: {similar_defect_ratio:.2%}")
            st.write(f"최근 7일 내 유사 불량 발생 횟수: {recent_similar_defect_count}")
            st.write(f"해당 그룹(mold_code): {main_group}")
            st.write(f"최근 발생 시각: {recent_time}")
            st.write(f"전체 생산 대비 비율: {similar_defect_rate_in_all:.2%}")
        else:
            st.info("표에서 불량 데이터를 선택하세요.")

    # --- 상단(우): 선택ID 시계열
    with col_main2:
        if sel_id is not None:
            sel_mold_code = origin_row['mold_code']
            sel_id = origin_row['id']
            df_mold = df[df['mold_code'] == sel_mold_code].sort_values('registration_time')
            sel_row_df = df_mold[df_mold['id'] == sel_id]
            tab1, tab2 = st.tabs(["금형 온도계열", "압력/슬리브/속도"])
            mold_temp_feats = ['upper_mold_temp1', 'lower_mold_temp1', 'lower_mold_temp2']
            for feat in mold_temp_feats:
                with tab1:
                    q_low, q_high = np.percentile(df_mold[feat].dropna(), [15, 85])
                    mask_iqr = (df_mold[feat] >= q_low) & (df_mold[feat] <= q_high)
                    df_trim = df_mold[mask_iqr]
                    x0 = df_trim[df_trim['passorfail'] == 0][feat].dropna().values
                    x1 = df_trim[df_trim['passorfail'] == 1][feat].dropna().values
                    split_val = kde_split_threshold(x0, x1)
                    fig = plot_timeseries_mold(df_mold, sel_row_df, feat, split_val)
                    st.plotly_chart(fig, use_container_width=True)
            press_feats = ['cast_pressure', 'sleeve_temperature', 'low_section_speed']
            for feat in press_feats:
                with tab2:
                    q_low, q_high = np.percentile(df_mold[feat].dropna(), [15, 85])
                    mask_iqr = (df_mold[feat] >= q_low) & (df_mold[feat] <= q_high)
                    df_trim = df_mold[mask_iqr]
                    x0 = df_trim[df_trim['passorfail'] == 0][feat].dropna().values
                    x1 = df_trim[df_trim['passorfail'] == 1][feat].dropna().values
                    split_val = kde_split_threshold(x0, x1)
                    fig = plot_timeseries_mold(df_mold, sel_row_df, feat, split_val, sel_color="green")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("좌측 표에서 불량 데이터를 선택하면 시계열 그래프가 표시됩니다.")

    # --- 중단1: SHAP 분석(좌)
    col_bottom1, col_bottom2 = st.columns([1, 2])
    with col_bottom1:
        shap_df = shap_results["df"]
        X = shap_results["X"]
        shap_values = shap_results["shap_values"]
        idx_list = shap_df.index[shap_df['id'].astype(str) == str(sel_id)].tolist()
        row_idx = idx_list[0] if len(idx_list) > 0 else 0
        n = 7
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
        contrib_show = contrib_show[::-1].reset_index(drop=True)
        colors = ['red' if v > 0 else 'blue' for v in contrib_show['SHAP_value']]
        max_abs = contrib_show['SHAP_value'].abs().max()
        fig = go.Figure(go.Bar(
            x=contrib_show["SHAP_value"],
            y=contrib_show["Feature"], orientation="h", marker_color=colors,
            text=[f"{v:+.3f}" for v in contrib_show["SHAP_value"]],
            textposition="auto", textfont=dict(color='black', size=13, family="Arial Black"),
            insidetextanchor='middle',
        ))
        fig.add_vline(x=0, line_color="gray", line_width=2)
        fig.update_traces(marker_line_color='black', marker_line_width=2)
        fig.update_layout(
            title="SHAP 변수별 기여도", xaxis_title="SHAP 값 (불량↑: +, 정상↑: -)", yaxis_title=None,
            height=300, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="#f5f5f5", font=dict(size=13),
        )
        fig.update_xaxes(range=[-max_abs * 1.1, max_abs * 1.1])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### SHAP 상위 변수")
        st.dataframe(
            contrib_show[::-1].reset_index(drop=True).style.format({"SHAP_value": "{:+.4f}"}),
            use_container_width=True, hide_index=True
        )

    # --- 중단2: 주요 변수 상관관계(우)
    with col_bottom2:
        main_feats = list(shap_results["feature_names"][shap_results["importances"].argsort()[::-1]][:10])
        all_feats = main_feats.copy()
        cross_corr = pd.DataFrame(index=main_feats, columns=all_feats)
        for main_feat in main_feats:
            for feat in all_feats:
                vals = X[[main_feat, feat]].dropna()
                v1, v2 = vals[main_feat], vals[feat]
                if isinstance(v1, pd.DataFrame): v1 = v1.iloc[:, 0]
                if isinstance(v2, pd.DataFrame): v2 = v2.iloc[:, 0]
                corr_val = v1.corr(v2) if len(v1) > 1 and len(v2) > 1 else np.nan
                cross_corr.loc[main_feat, feat] = corr_val
        cross_corr = cross_corr.astype(float)
        cross_corr_no_diag = cross_corr.copy()
        for i in range(len(main_feats)):
            if main_feats[i] in all_feats:
                cross_corr_no_diag.iloc[i, all_feats.index(main_feats[i])] = 0
        flat_corr = [
            (main_feats[i], main_feats[j], cross_corr_no_diag.values[i, j])
            for i in range(len(main_feats))
            for j in range(len(main_feats))
            if i < j
        ]
        top3 = [x for x in sorted(flat_corr, key=lambda x: abs(x[2]), reverse=True) if not np.isnan(x[2])][:3]
        show_top3 = st.checkbox("상위 3 상관관계 변수쌍 보기", value=False)
        placeholder = st.empty()
        if show_top3:
            items = [
                f"<span style='font-size:18px'>• <b>{m}</b> ↔ <b>{a}</b> : {v:+.2f}</span>"
                for m, a, v in top3
            ]
            placeholder.markdown("<br>".join(items), unsafe_allow_html=True)
        else:
            placeholder.markdown("<br>" * len(top3), unsafe_allow_html=True)

        fig_corr = go.Figure(data=go.Heatmap(
            z=cross_corr.values,
            x=main_feats,
            y=main_feats,
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(cross_corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="상관계수", tickfont=dict(size=12))
        ))
        # 상위 3개 상관관계쌍 테두리 강조
        for m, a, v in top3:
            y_idx = main_feats.index(m)
            x_idx = main_feats.index(a)
            fig_corr.add_shape(
                type="rect",
                x0=x_idx - 0.5, x1=x_idx + 0.5,
                y0=y_idx - 0.5, y1=y_idx + 0.5,
                line=dict(color="green", width=4),
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

    # ---------------------- 하단 권장구간/박스플롯 -----------------------
    main_features = list(shap_results["feature_names"][shap_results["importances"].argsort()[::-1]][:5])
    split_dict = {}
    split_bins = {}
    for feature in main_features:
        q_low, q_high = np.percentile(df[feature].dropna(), [15, 85])
        mask_iqr = (df[feature] >= q_low) & (df[feature] <= q_high)
        df_trim = df[mask_iqr]
        x0 = df_trim[df_trim['passorfail'] == 0][feature].dropna().values
        x1 = df_trim[df_trim['passorfail'] == 1][feature].dropna().values
        thr = kde_split_threshold(x0, x1)
        if thr is not None:
            split_dict[feature] = [thr]
            split_bins[feature] = [-np.inf, thr, np.inf]
        else:
            split_dict[feature] = []
            split_bins[feature] = [-np.inf, np.inf]

    col_extra1, col_extra2 = st.columns([1, 2])
    with col_extra1:
        st.markdown("#### 주요 변수 권장구간")
        summary_msgs = []
        for feature in main_features:
            bins = split_bins[feature]
            y_true = df['passorfail']
            bin_rates = []
            for i in range(len(bins) - 1):
                mask = (df[feature] > bins[i]) & (df[feature] <= bins[i + 1])
                rate = y_true[mask].mean() if mask.sum() > 0 else np.nan
                bin_rates.append(rate)
            min_idx = np.nanargmin(bin_rates)
            if len(bins) == 3:
                ranges = [f"{bins[1]:.2f} 이하", f"{bins[1]:.2f} 이상"]
            else:
                ranges = ["전체 구간"]
            summary_msgs.append(
                f"**{feature}** : <b>{ranges[min_idx]}</b> 유지 권장<br>"
                + " / ".join(
                    f"[{r}] {br:.2%}" if not np.isnan(br) else f"[{r}] N/A"
                    for r, br in zip(ranges, bin_rates)
                )
            )
        st.markdown("<br><br>".join(summary_msgs), unsafe_allow_html=True)
    with col_extra2:
        st.markdown("#### 주요 변수 박스플랏")
        tab_names = main_features
        tab = st.tabs(tab_names)
        for i, feature in enumerate(main_features):
            with tab[i]:
                vals_0 = df[df['passorfail'] == 0][feature].dropna()
                vals_1 = df[df['passorfail'] == 1][feature].dropna()
                thrs = split_dict[feature]
                fig = plot_boxplot_with_thres(vals_0, vals_1, feature, thrs[0] if thrs else None)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()