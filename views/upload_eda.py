"""
Trang Upload & EDA - Upload dữ liệu và phân tích khám phá
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from utils.ui_components import show_llm_analysis, show_processing_placeholder
from utils.session_state import init_session_state, clear_data_related_state
from backend.llm_integration import analyze_eda_with_llm, get_eda_summary, LLMConfig


def render_eda_tabs(data, key_suffix=""):
    """Helper function to render EDA tabs - reusable for both uploaded and cached data"""
    
    # Import inside function to avoid circular imports
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Dữ Liệu Mẫu", 
        "📊 Thống Kê Mô Tả", 
        "📈 Phân Phối Dữ Liệu",
        "🤖 Phân Tích AI"
    ])
    
    # The tabs will be rendered by the main render function
    return tab1, tab2, tab3, tab4


def render():
    """Render trang Upload & EDA"""
    print("DEBUG: Starting upload_eda.render()")
    try:
        init_session_state()
        print("DEBUG: Session state initialized")
    except Exception as e:
        st.error(f"Error initializing session: {e}")
        print(f"ERROR: Session init failed: {e}")
        return
    
    st.markdown("## 📤 Upload Dữ Liệu & Phân Tích Khám Phá (EDA)")
    st.markdown("Tải lên file CSV chứa dữ liệu khách hàng và khám phá các thông tin quan trọng.")
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Chọn file CSV dữ liệu",
        type=['csv'],
        help="Upload file CSV chứa dữ liệu khách hàng với các đặc trưng và nhãn",
        key="csv_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Check if this is a new file
            uploaded_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            is_new_file = st.session_state.get('current_file_id') != uploaded_file_id
            
            # Load data with error handling
            data = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='utf-8')
            
            # Validate data
            if data.empty:
                st.error("❌ File is empty or invalid format")
                return
            
            if len(data) < 5:
                st.warning(f"⚠️ Dataset only has {len(data)} rows. Upload more data for better analysis.")
            
            # Only clear state if this is a NEW file
            if is_new_file:
                clear_data_related_state()
                st.session_state.current_file_id = uploaded_file_id
                st.info("🔄 File mới được tải lên - Đã xóa các cấu hình cũ")
            
            st.session_state.data = data
            st.success(f"✅ Data loaded successfully! ({len(data)} rows, {len(data.columns)} columns)")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Dữ Liệu Mẫu", 
                "📊 Thống Kê Mô Tả", 
                "📈 Phân Phối Dữ Liệu",
                "🤖 Phân Tích AI"
            ])
            
            # Tab 1: Sample Data
            with tab1:
                st.markdown("### 📋 Dữ Liệu Mẫu")
                
                # Controls
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"📊 Hiển thị toàn bộ {len(data):,} dòng dữ liệu")
                with col2:
                    show_charts = st.checkbox("Hiện biểu đồ", value=True, key="show_charts")
                
                # Use all data
                display_data = data.copy()
                
                # Show charts header if enabled
                if show_charts:
                    st.markdown("---")
                    
                    # Generate mini charts as base64 images for each column
                    import base64
                    from io import BytesIO
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    # Create header row with visualizations
                    header_html = "<div style='overflow-x: auto;'><table style='width: 100%; border-collapse: collapse; font-size: 0.85rem;'>"
                    
                    # Header row with charts
                    header_html += "<tr style='background-color: #1e1e1e;'>"
                    
                    for col_name in data.columns:
                        col_data = data[col_name]
                        header_html += f"<td style='border: 1px solid #444; padding: 10px; text-align: center; vertical-align: top; min-width: 120px;'>"
                        header_html += f"<div style='font-weight: bold; margin-bottom: 5px;'>{col_name}</div>"
                        
                        # Generate chart
                        if pd.api.types.is_numeric_dtype(col_data):
                            # Numeric - Histogram
                            col_clean = col_data.dropna()
                            if len(col_clean) > 0:
                                fig, ax = plt.subplots(figsize=(1.5, 0.8), facecolor='none')
                                ax.hist(col_clean, bins=min(15, max(5, len(col_clean) // 10)), color='#667eea', edgecolor='none')
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                ax.spines['bottom'].set_visible(False)
                                ax.spines['left'].set_visible(False)
                                ax.patch.set_alpha(0)
                                
                                # Save to base64
                                buffer = BytesIO()
                                plt.savefig(buffer, format='png', bbox_inches='tight', transparent=True, dpi=50)
                                buffer.seek(0)
                                img_base64 = base64.b64encode(buffer.read()).decode()
                                plt.close(fig)
                                
                                header_html += f"<img src='data:image/png;base64,{img_base64}' style='width: 100%; max-width: 120px;'/>"
                                header_html += f"<div style='font-size: 0.7rem; margin-top: 3px;'>Min: {col_clean.min():.1f} | Max: {col_clean.max():.1f}</div>"
                                header_html += f"<div style='font-size: 0.7rem;'>Mean: {col_clean.mean():.1f} | Unique: {col_data.nunique()}</div>"
                        else:
                            # Categorical - Bar chart
                            value_counts = col_data.value_counts().head(3)
                            total = len(col_data)
                            
                            if len(value_counts) > 0:
                                percentages = (value_counts / total * 100)
                                
                                fig, ax = plt.subplots(figsize=(1.5, 0.8), facecolor='none')
                                ax.barh(range(len(value_counts)), percentages.values, color='#764ba2')
                                ax.set_yticks(range(len(value_counts)))
                                ax.set_yticklabels([str(v)[:8] for v in value_counts.index], fontsize=6, color='white')
                                ax.set_xticks([])
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                ax.spines['bottom'].set_visible(False)
                                ax.spines['left'].set_visible(False)
                                ax.patch.set_alpha(0)
                                ax.invert_yaxis()
                                
                                # Add percentage labels
                                for i, (idx, pct) in enumerate(zip(value_counts.index, percentages.values)):
                                    ax.text(pct + 2, i, f'{pct:.0f}%', va='center', fontsize=6, color='white')
                                
                                buffer = BytesIO()
                                plt.savefig(buffer, format='png', bbox_inches='tight', transparent=True, dpi=50)
                                buffer.seek(0)
                                img_base64 = base64.b64encode(buffer.read()).decode()
                                plt.close(fig)
                                
                                header_html += f"<img src='data:image/png;base64,{img_base64}' style='width: 100%; max-width: 120px;'/>"
                                header_html += f"<div style='font-size: 0.7rem; margin-top: 3px;'>Unique: {col_data.nunique()} | Mode: {str(value_counts.index[0])[:10]}</div>"
                        
                        # Missing info
                        missing_count = col_data.isnull().sum()
                        missing_pct = (missing_count / len(col_data) * 100) if len(col_data) > 0 else 0
                        if missing_count > 0:
                            header_html += f"<div style='font-size: 0.65rem; color: #ffaa00; margin-top: 2px;'>⚠️ Missing: {missing_count} ({missing_pct:.1f}%)</div>"
                        else:
                            header_html += f"<div style='font-size: 0.65rem; color: #44ff44; margin-top: 2px;'>✅ No missing</div>"
                        
                        header_html += "</td>"
                    
                    header_html += "</tr></table></div>"
                    
                    # Display header with charts
                    st.markdown(header_html, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Display the dataframe with pagination
                st.dataframe(display_data, use_container_width=True, height=500)
                
                # Data info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Tổng số dòng", f"{len(data):,}")
                with col2:
                    st.metric("📋 Tổng số cột", len(data.columns))
                with col3:
                    missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
                    st.metric("❓ Dữ liệu thiếu", f"{missing_pct:.1f}%")
                with col4:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    st.metric("🔢 Cột số", len(numeric_cols))
            
            # Tab 2: Descriptive Statistics
            with tab2:
                st.markdown("### 📊 Thống Kê Mô Tả")
                
                # Numeric columns stats
                numeric_data = data.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    st.markdown("#### 🔢 Biến Số")
                    
                    stats_df = numeric_data.describe().T
                    stats_df['missing'] = data[numeric_data.columns].isnull().sum()
                    stats_df['missing_pct'] = (stats_df['missing'] / len(data) * 100).round(2)
                    
                    # Highlight styling
                    st.dataframe(
                        stats_df.style.background_gradient(cmap='viridis', subset=['mean', 'std']),
                        use_container_width=True
                    )
                    
                    # Download stats
                    csv = stats_df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        "📥 Tải Thống Kê (CSV)",
                        csv,
                        "statistics.csv",
                        "text/csv",
                        key='download-stats'
                    )
                    
                    st.markdown("---")
                    
                    # Detailed column analysis
                    st.markdown("#### 🔍 Phân Tích Chi Tiết Từng Cột")
                    
                    numeric_cols = numeric_data.columns.tolist()
                    selected_numeric_col = st.selectbox(
                        "Chọn cột số để phân tích chi tiết:",
                        numeric_cols,
                        key="detailed_numeric_col"
                    )
                    
                    if selected_numeric_col:
                        st.markdown(f"### 📊 Dashboard Phân Tích: `{selected_numeric_col}`")
                        
                        col_data = data[selected_numeric_col].dropna()
                        
                        # Summary metrics
                        st.markdown("#### 📈 Tóm Tắt Thống Kê")
                        metric_cols = st.columns(6)
                        
                        with metric_cols[0]:
                            st.metric("Count", f"{len(col_data):,}")
                        with metric_cols[1]:
                            st.metric("Mean", f"{col_data.mean():.2f}")
                        with metric_cols[2]:
                            st.metric("Median", f"{col_data.median():.2f}")
                        with metric_cols[3]:
                            st.metric("Std Dev", f"{col_data.std():.2f}")
                        with metric_cols[4]:
                            st.metric("Min", f"{col_data.min():.2f}")
                        with metric_cols[5]:
                            st.metric("Max", f"{col_data.max():.2f}")
                        
                        st.markdown("---")
                        
                        # Charts section
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            # Histogram with default bins
                            st.markdown("##### 📊 Histogram & Distribution")
                            fig_hist = px.histogram(
                                data,
                                x=selected_numeric_col,
                                marginal="box",
                                color_discrete_sequence=['#667eea']
                            )
                            fig_hist.update_layout(
                                template="plotly_dark",
                                height=350,
                                showlegend=False,
                                xaxis_title=selected_numeric_col,
                                yaxis_title="Frequency"
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with chart_col2:
                            # Box plot for outlier detection
                            st.markdown("##### 📦 Box Plot (Outlier Detection)")
                            fig_box = go.Figure()
                            fig_box.add_trace(go.Box(
                                y=col_data,
                                name=selected_numeric_col,
                                boxmean='sd',
                                marker_color='#764ba2',
                                boxpoints='outliers'
                            ))
                            fig_box.update_layout(
                                template="plotly_dark",
                                height=350,
                                showlegend=False,
                                yaxis_title=selected_numeric_col
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                        
                        # Quantile and outlier analysis
                        st.markdown("---")
                        stat_col1, stat_col2 = st.columns(2)
                        
                        with stat_col1:
                            st.markdown("##### 📊 Phân Vị (Quantiles)")
                            quantiles = col_data.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
                            quantile_df = pd.DataFrame({
                                'Phân vị': ['1%', '5%', '25%', '50% (Median)', '75%', '95%', '99%'],
                                'Giá trị': quantiles.values
                            })
                            st.dataframe(
                                quantile_df.style.format({'Giá trị': '{:.2f}'}),
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        with stat_col2:
                            st.markdown("##### ⚠️ Outlier Analysis (IQR Method)")
                            Q1 = col_data.quantile(0.25)
                            Q3 = col_data.quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                            outlier_pct = (len(outliers) / len(col_data) * 100)
                            
                            outlier_info = pd.DataFrame({
                                'Metric': ['Lower Bound', 'Upper Bound', 'Số Outliers', 'Tỷ lệ Outliers'],
                                'Value': [
                                    f"{lower_bound:.2f}",
                                    f"{upper_bound:.2f}",
                                    f"{len(outliers):,}",
                                    f"{outlier_pct:.2f}%"
                                ]
                            })
                            st.dataframe(outlier_info, use_container_width=True, hide_index=True)
                        
                        # Distribution characteristics
                        st.markdown("---")
                        st.markdown("##### 📐 Đặc Điểm Phân Phối")
                        
                        dist_cols = st.columns(4)
                        
                        # Skewness
                        skewness = stats.skew(col_data)
                        with dist_cols[0]:
                            st.metric("Skewness", f"{skewness:.3f}")
                            if abs(skewness) < 0.5:
                                st.caption("✅ Gần đối xứng")
                            elif skewness > 0:
                                st.caption("➡️ Lệch phải")
                            else:
                                st.caption("⬅️ Lệch trái")
                        
                        # Kurtosis
                        kurtosis = stats.kurtosis(col_data)
                        with dist_cols[1]:
                            st.metric("Kurtosis", f"{kurtosis:.3f}")
                            if abs(kurtosis) < 0.5:
                                st.caption("✅ Phân phối chuẩn")
                            elif kurtosis > 0:
                                st.caption("📈 Nhọn (peaked)")
                            else:
                                st.caption("📉 Bẹt (flat)")
                        
                        # Range
                        with dist_cols[2]:
                            st.metric("Range", f"{col_data.max() - col_data.min():.2f}")
                            st.caption("Max - Min")
                        
                        # CV (Coefficient of Variation)
                        cv = (col_data.std() / col_data.mean() * 100) if col_data.mean() != 0 else 0
                        with dist_cols[3]:
                            st.metric("CV", f"{cv:.2f}%")
                            if cv < 15:
                                st.caption("✅ Độ biến thiên thấp")
                            elif cv < 30:
                                st.caption("⚠️ Độ biến thiên trung bình")
                            else:
                                st.caption("🔴 Độ biến thiên cao")
                        
                        # Value distribution table
                        st.markdown("---")
                        st.markdown("##### 📋 Phân Bổ Giá Trị (Binned)")
                        
                        # Bins slider for binned distribution
                        bin_slider_col1, bin_slider_col2 = st.columns([3, 1])
                        with bin_slider_col1:
                            n_bins = st.slider(
                                f"Số bins cho {selected_numeric_col}:",
                                min_value=1,
                                max_value=20,
                                value=10,
                                step=1,
                                key=f"binned_dist_{selected_numeric_col}"
                            )
                        
                        # Create bins
                        bins = pd.cut(col_data, bins=n_bins)
                        bin_counts = bins.value_counts().sort_index()
                        
                        bin_df = pd.DataFrame({
                            'Khoảng giá trị': bin_counts.index.astype(str),
                            'Số lượng': bin_counts.values,
                            'Tỷ lệ (%)': (bin_counts.values / len(col_data) * 100).round(2)
                        })
                        
                        st.dataframe(bin_df, use_container_width=True, hide_index=True)
                        
                        # Histogram of bins
                        fig_bin = px.bar(
                            bin_df,
                            x='Khoảng giá trị',
                            y='Số lượng',
                            color='Tỷ lệ (%)',
                            color_continuous_scale='Viridis',
                            title=f"Phân bổ giá trị của {selected_numeric_col}"
                        )
                        fig_bin.update_layout(
                            template="plotly_dark",
                            height=350,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_bin, use_container_width=True)
                
                # Categorical columns
                categorical_data = data.select_dtypes(include=['object', 'category'])
                if not categorical_data.empty:
                    st.markdown("#### 📝 Biến Phân Loại")
                    
                    cat_info = []
                    for col in categorical_data.columns:
                        cat_info.append({
                            'Tên cột': col,
                            'Số giá trị khác nhau': data[col].nunique(),
                            'Giá trị phổ biến nhất': data[col].mode()[0] if not data[col].mode().empty else 'N/A',
                            'Tần suất cao nhất': data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0,
                            'Thiếu': data[col].isnull().sum(),
                            'Tỷ lệ thiếu (%)': f"{data[col].isnull().sum() / len(data) * 100:.2f}"
                        })
                    
                    cat_df = pd.DataFrame(cat_info)
                    st.dataframe(cat_df, use_container_width=True)
            
            # Tab 3: Data Distribution
            with tab3:
                st.markdown("### 📈 Phân Phối & Tương Quan Dữ Liệu")
                
                viz_type = st.radio(
                    "Chọn loại phân tích:",
                    ["Correlation Heatmap", "Scatter Plot Matrix", "Scatter Plot (2 Biến)", "Grouped Analysis"],
                    horizontal=True
                )
                
                if viz_type == "Correlation Heatmap":
                    st.markdown("#### � Ma Trận Tương Quan")
                    
                    numeric_data = data.select_dtypes(include=[np.number])
                    if not numeric_data.empty and len(numeric_data.columns) > 1:
                        corr_matrix = numeric_data.corr()
                        
                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            text_auto='.2f',
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="Ma trận tương quan giữa các biến",
                            zmin=-1,
                            zmax=1
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Find high correlations
                        st.markdown("#### 🔍 Các Cặp Biến Có Tương Quan Cao")
                        
                        threshold = st.slider("Ngưỡng tương quan:", 0.5, 0.95, 0.7, 0.05, key="upload_corr_threshold")
                        
                        high_corr = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                if abs(corr_matrix.iloc[i, j]) >= threshold:
                                    high_corr.append({
                                        'Biến 1': corr_matrix.columns[i],
                                        'Biến 2': corr_matrix.columns[j],
                                        'Tương quan': f"{corr_matrix.iloc[i, j]:.3f}",
                                        'Loại': 'Dương' if corr_matrix.iloc[i, j] > 0 else 'Âm'
                                    })
                        
                        if high_corr:
                            st.dataframe(pd.DataFrame(high_corr), use_container_width=True, hide_index=True)
                        else:
                            st.info(f"Không tìm thấy cặp biến nào có tương quan >= {threshold}")
                    else:
                        st.warning("Cần ít nhất 2 biến số để tạo ma trận tương quan.")
                
                elif viz_type == "Scatter Plot Matrix":
                    st.markdown("#### 🔷 Scatter Plot Matrix (Pair Plot)")
                    st.caption("Hiển thị mối quan hệ giữa từng cặp biến số")
                    
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) >= 2:
                        # Allow selection of variables
                        max_vars = min(5, len(numeric_cols))
                        selected_vars = st.multiselect(
                            "Chọn các biến để hiển thị (tối đa 5):",
                            numeric_cols,
                            default=numeric_cols[:max_vars],
                            max_selections=5,
                            key="upload_scatter_matrix_vars"
                        )
                        
                        if len(selected_vars) >= 2:
                            # Create scatter matrix
                            fig = px.scatter_matrix(
                                data,
                                dimensions=selected_vars,
                                color_discrete_sequence=['#667eea'],
                                opacity=0.6
                            )
                            
                            fig.update_layout(
                                template="plotly_dark",
                                height=800,
                                title="Scatter Plot Matrix - Phân tích quan hệ từng cặp biến"
                            )
                            
                            fig.update_traces(diagonal_visible=False, showupperhalf=False)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info("💡 **Gợi ý**: Tìm kiếm các pattern tuyến tính hoặc phi tuyến giữa các cặp biến.")
                        else:
                            st.warning("Vui lòng chọn ít nhất 2 biến.")
                    else:
                        st.warning("Cần ít nhất 2 biến số để tạo Scatter Plot Matrix.")
                
                elif viz_type == "Scatter Plot (2 Biến)":
                    st.markdown("#### � Phân Tích Chi Tiết 2 Biến")
                    
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            x_var = st.selectbox("Chọn biến X:", numeric_cols, key="upload_scatter_x")
                        with col2:
                            y_vars = [col for col in numeric_cols if col != x_var]
                            y_var = st.selectbox("Chọn biến Y:", y_vars, key="upload_scatter_y")
                        
                        # Options
                        opt_col1, opt_col2, opt_col3 = st.columns(3)
                        with opt_col1:
                            show_trendline = st.checkbox("Hiện đường xu hướng", value=True, key="upload_scatter_trend")
                        with opt_col2:
                            show_marginal = st.checkbox("Hiện phân phối biên", value=True, key="upload_scatter_marginal")
                        with opt_col3:
                            color_by_cat = st.checkbox("Tô màu theo biến phân loại", value=False, key="upload_scatter_color")
                        
                        # Color selection
                        color_var = None
                        if color_by_cat:
                            cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                            if cat_cols:
                                color_var = st.selectbox("Chọn biến phân loại:", cat_cols, key="upload_scatter_color_var")
                        
                        # Create scatter plot
                        fig = px.scatter(
                            data,
                            x=x_var,
                            y=y_var,
                            color=color_var,
                            trendline="ols" if show_trendline else None,
                            marginal_x="histogram" if show_marginal else None,
                            marginal_y="histogram" if show_marginal else None,
                            opacity=0.6,
                            title=f"Mối quan hệ giữa {x_var} và {y_var}"
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate correlation
                        corr = data[x_var].corr(data[y_var])
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("Tương quan Pearson", f"{corr:.3f}")
                        with metric_col2:
                            if abs(corr) >= 0.7:
                                st.metric("Mức độ", "Mạnh 💪", delta="Tương quan cao")
                            elif abs(corr) >= 0.4:
                                st.metric("Mức độ", "Trung bình ⚖️", delta="Tương quan vừa")
                            else:
                                st.metric("Mức độ", "Yếu 📉", delta="Tương quan thấp")
                        with metric_col3:
                            st.metric("Loại", "Dương ↗️" if corr > 0 else "Âm ↘️")
                    else:
                        st.warning("Cần ít nhất 2 biến số.")
                
                else:  # Grouped Analysis
                    st.markdown("#### 📦 Phân Tích Theo Nhóm")
                    st.caption("So sánh phân phối biến số theo các nhóm phân loại")
                    
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if numeric_cols and cat_cols:
                        col1, col2 = st.columns(2)
                        with col1:
                            num_var = st.selectbox("Chọn biến số:", numeric_cols, key="upload_group_num")
                        with col2:
                            cat_var = st.selectbox("Chọn biến phân loại:", cat_cols, key="upload_group_cat")
                        
                        # Limit categories to avoid clutter
                        unique_cats = data[cat_var].nunique()
                        if unique_cats > 10:
                            st.warning(f"⚠️ Biến {cat_var} có {unique_cats} nhóm. Chỉ hiển thị 10 nhóm phổ biến nhất.")
                            top_cats = data[cat_var].value_counts().head(10).index
                            plot_data = data[data[cat_var].isin(top_cats)]
                        else:
                            plot_data = data
                        
                        # Choose plot type
                        plot_type = st.radio(
                            "Loại biểu đồ:",
                            ["Box Plot", "Violin Plot", "Strip Plot"],
                            horizontal=True,
                            key="upload_group_plot_type"
                        )
                        
                        if plot_type == "Box Plot":
                            fig = px.box(
                                plot_data,
                                x=cat_var,
                                y=num_var,
                                color=cat_var,
                                title=f"Phân phối {num_var} theo {cat_var}",
                                points="outliers"
                            )
                        elif plot_type == "Violin Plot":
                            fig = px.violin(
                                plot_data,
                                x=cat_var,
                                y=num_var,
                                color=cat_var,
                                title=f"Phân phối {num_var} theo {cat_var}",
                                box=True,
                                points="outliers"
                            )
                        else:  # Strip Plot
                            fig = px.strip(
                                plot_data,
                                x=cat_var,
                                y=num_var,
                                color=cat_var,
                                title=f"Phân phối {num_var} theo {cat_var}"
                            )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=500,
                            xaxis_tickangle=-45
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics by group
                        st.markdown("#### 📊 Thống Kê Theo Nhóm")
                        group_stats = plot_data.groupby(cat_var)[num_var].agg([
                            ('Số lượng', 'count'),
                            ('Trung bình', 'mean'),
                            ('Trung vị', 'median'),
                            ('Độ lệch chuẩn', 'std'),
                            ('Min', 'min'),
                            ('Max', 'max')
                        ]).round(2)
                        
                        st.dataframe(group_stats, use_container_width=True)
                    else:
                        if not numeric_cols:
                            st.warning("Không có biến số nào trong dữ liệu.")
                        if not cat_cols:
                            st.warning("Không có biến phân loại nào trong dữ liệu.")
            
            # Tab 4: AI Analysis
            with tab4:
                st.markdown("### 🤖 Phân Tích Tự Động Bằng AI")
                
                # Check LLM configuration
                is_llm_configured = LLMConfig.is_configured()
                
                if not is_llm_configured:
                    st.info("""
                    ℹ️ **Chưa cấu hình LLM API**
                    
                    Để sử dụng phân tích AI chi tiết, vui lòng:
                    1. Tạo file `.env` trong thư mục gốc
                    2. Thêm Google API key: `GOOGLE_API_KEY=...`
                    3. (Tùy chọn) Chọn model: `GOOGLE_MODEL=gemini-2.5-flash`
                    4. (Tùy chọn) Chọn provider: `LLM_PROVIDER=google`
                    
                    **Lấy Google API key miễn phí tại: https://aistudio.google.com/app/apikey**
                    
                    **Hiện tại sẽ sử dụng chế độ phân tích tự động cơ bản.**
                    """)
                
                st.markdown("""
                <div style="background-color: #262730; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                    <h4 style="margin-top: 0; color: #667eea;">💡 Phân Tích Tự Động</h4>
                    <p>AI sẽ phân tích toàn bộ kết quả EDA và cung cấp:</p>
                    <ul>
                        <li>✨ Đánh giá chất lượng dữ liệu tổng thể</li>
                        <li>📊 Nhận xét về phân phối các biến quan trọng</li>
                        <li>🔗 Phát hiện tương quan và mối quan hệ giữa các biến</li>
                        <li>⚠️ Cảnh báo về outliers, missing data và vấn đề tiềm ẩn</li>
                        <li>💡 Đề xuất roadmap tiền xử lý dữ liệu chi tiết</li>
                        <li>🎯 Dự đoán khả năng xây dựng mô hình hiệu quả</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Options
                col1, col2 = st.columns([3, 1])
                with col1:
                    analysis_button = st.button(
                        "🔄 Tạo Phân Tích AI" if is_llm_configured else "📊 Tạo Phân Tích Tự Động",
                        use_container_width=True,
                        type="primary",
                        key="ai_analysis_btn"
                    )
                with col2:
                    show_raw_summary = st.checkbox("Xem EDA Summary", value=False, key="show_eda_raw")
                
                # Show raw EDA summary if requested
                if show_raw_summary:
                    st.markdown("---")
                    st.markdown("#### 📋 EDA Summary (Raw Data)")
                    with st.expander("Xem dữ liệu thống kê chi tiết", expanded=False):
                        summary_text = get_eda_summary(data, format="text")
                        st.text(summary_text)
                
                # Generate AI analysis
                if analysis_button:
                    with st.spinner("🤖 Đang phân tích dữ liệu..." if is_llm_configured else "📊 Đang tạo báo cáo..."):
                        try:
                            # Get API key and provider from config
                            api_key = LLMConfig.get_api_key() if is_llm_configured else None
                            provider = LLMConfig.DEFAULT_PROVIDER
                            
                            # Analyze with LLM
                            analysis_result = analyze_eda_with_llm(data, api_key=api_key, provider=provider)
                            
                            # Store in session state
                            st.session_state.ai_analysis = analysis_result
                            
                            st.success("✅ Phân tích hoàn thành!" if is_llm_configured else "✅ Báo cáo đã được tạo!")
                        
                        except Exception as e:
                            st.error(f"❌ Lỗi khi tạo phân tích: {str(e)}")
                            st.info("💡 Vui lòng kiểm tra API key và kết nối internet.")
                            import traceback
                            st.code(traceback.format_exc())
                
                # Display analysis if available
                if 'ai_analysis' in st.session_state and st.session_state.ai_analysis:
                    st.markdown("---")
                    st.markdown("### 📝 Kết Quả Phân Tích")
                    
                    # Display in a nice container
                    with st.container():
                        st.markdown(st.session_state.ai_analysis)
                    
                    # Auto-generate preprocessing suggestions (but don't display here)
                    # Check if we already have suggestions
                    if 'preprocessing_suggestions' not in st.session_state:
                        # Auto-generate on first time
                        with st.spinner("🤖 Đang tạo gợi ý tiền xử lý từ AI..."):
                            try:
                                # Call LLM to generate preprocessing suggestions
                                provider = LLMConfig.DEFAULT_PROVIDER
                                api_key = LLMConfig.get_api_key() if is_llm_configured else None
                                
                                # Create prompt for preprocessing suggestions
                                suggestions_prompt = f"""Dựa trên kết quả phân tích EDA sau đây, hãy tạo một roadmap tiền xử lý dữ liệu theo ĐÚNG 4 BƯỚC sau:

KẾT QUẢ PHÂN TÍCH EDA:
{st.session_state.ai_analysis}

YÊU CẦU:
Trả về roadmap tiền xử lý theo ĐÚNG 4 BƯỚC SAU (không được thêm bớt bước):

**Bước 1: Xử Lý Biến Định Danh & Giá Trị Không Hợp Lệ**
- Xác định và loại bỏ các cột định danh (ID, customer_id, ...)
- Phát hiện và xử lý các giá trị không hợp lệ (âm, ngoài phạm vi, ...)
- Liệt kê CỤ THỂ tên cột cần xử lý

**Bước 2: Xử Lý Giá Trị Thiếu**
- Xác định các cột có missing values
- Đề xuất phương pháp xử lý CHO TỪNG CỘT (drop, imputation, ...)
- Giải thích lý do chọn phương pháp đó

**Bước 3: Xử Lý Outliers & Biến Đổi Phân Phối**
- Xác định các cột có outliers nghiêm trọng
- Đề xuất phương pháp xử lý outliers (Winsorization, IQR, Log transform, ...)
- Đề xuất biến đổi phân phối nếu cần (Log, Box-Cox, ...)

**Bước 4: Mã Hóa Biến Phân Loại**
- Xác định các biến phân loại cần mã hóa
- Đề xuất phương pháp mã hóa CHO TỪNG CỘT (One-Hot, Label Encoding, Target Encoding, ...)
- Giải thích lý do chọn phương pháp đó (dựa vào cardinality, mối quan hệ với target, ...)

FORMAT:
- Mỗi bước phải có tiêu đề in đậm với emoji
- Dưới mỗi bước là danh sách gạch đầu dòng CHI TIẾT, CỤ THỂ
- Đề cập TÊN CỘT và PHƯƠNG PHÁP cụ thể
- Ngôn ngữ: Tiếng Việt chuyên nghiệp

QUAN TRỌNG: 
- PHẢI trả về ĐÚNG 4 BƯỚC theo cấu trúc trên
- Mỗi bước phải CỤ THỂ, đề cập tên cột và phương pháp
- KHÔNG thêm bước khác, KHÔNG tóm tắt chung chung
- CHỈ trả về 4 bước, KHÔNG giải thích thêm!"""

                                # Call LLM
                                if is_llm_configured and api_key:
                                    if provider == "google":
                                        import google.generativeai as genai
                                        genai.configure(api_key=api_key)
                                        model = genai.GenerativeModel(LLMConfig.get_model(provider))
                                        response = model.generate_content(suggestions_prompt)
                                        suggestions_text = response.text.strip()
                                    else:
                                        # Fallback for other providers or no API
                                        suggestions_text = """**📋 Roadmap Tiền Xử Lý Dữ Liệu:**

**Bước 1: 🔍 Xử Lý Biến Định Danh & Giá Trị Không Hợp Lệ**
- Xác định và loại bỏ các cột định danh (customer_id, ID, ...)
- Kiểm tra và xử lý các giá trị không hợp lệ (âm, ngoài phạm vi hợp lý)

**Bước 2: ❓ Xử Lý Giá Trị Thiếu**
- Xác định các cột có missing values
- Áp dụng phương pháp phù hợp: Drop, Mean/Median Imputation, hoặc Forward Fill

**Bước 3: ⚠️ Xử Lý Outliers & Biến Đổi Phân Phối**
- Phát hiện outliers bằng phương pháp IQR hoặc Z-score
- Áp dụng Winsorization hoặc Log Transform cho các cột có outliers
- Biến đổi phân phối lệch bằng Log hoặc Box-Cox nếu cần

**Bước 4: 🔤 Mã Hóa Biến Phân Loại**
- One-Hot Encoding cho biến có cardinality thấp (< 10 categories)
- Label Encoding cho biến ordinal hoặc binary
- Target Encoding cho biến có cardinality cao"""
                                else:
                                    suggestions_text = """**📋 Roadmap Tiền Xử Lý Dữ Liệu:**

**Bước 1: 🔍 Xử Lý Biến Định Danh & Giá Trị Không Hợp Lệ**
- Xác định và loại bỏ các cột định danh (customer_id, ID, ...)
- Kiểm tra và xử lý các giá trị không hợp lệ (âm, ngoài phạm vi hợp lý)

**Bước 2: ❓ Xử Lý Giá Trị Thiếu**
- Xác định các cột có missing values
- Áp dụng phương pháp phù hợp: Drop, Mean/Median Imputation, hoặc Forward Fill

**Bước 3: ⚠️ Xử Lý Outliers & Biến Đổi Phân Phối**
- Phát hiện outliers bằng phương pháp IQR hoặc Z-score
- Áp dụng Winsorization hoặc Log Transform cho các cột có outliers
- Biến đổi phân phối lệch bằng Log hoặc Box-Cox nếu cần

**Bước 4: 🔤 Mã Hóa Biến Phân Loại**
- One-Hot Encoding cho biến có cardinality thấp (< 10 categories)
- Label Encoding cho biến ordinal hoặc binary
- Target Encoding cho biến có cardinality cao"""
                                
                                # Save to session state (silently, no notification)
                                st.session_state.preprocessing_suggestions = suggestions_text
                                st.session_state.eda_analysis_result = st.session_state.ai_analysis
                                st.session_state.llm_provider = provider
                                
                            except Exception as e:
                                # Save error message but don't show notification
                                st.session_state.preprocessing_suggestions = f"⚠️ Không thể tạo gợi ý tự động: {str(e)}\n\nVui lòng xem phân tích EDA ở trên để tự đưa ra các bước tiền xử lý."
                    
                    # Download option
                    st.markdown("---")
                    st.download_button(
                        label="📥 Tải xuống phân tích (Markdown)",
                        data=st.session_state.ai_analysis,
                        file_name="eda_analysis.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                else:
                    st.markdown("---")
                    st.info("👆 Nhấn nút phía trên để bắt đầu phân tích!")
        
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file: {str(e)}")
    
    else:
        # If data exists, show full EDA with option to clear
        if 'data' in st.session_state and st.session_state.data is not None:
            data = st.session_state.data
            
            # Show info bar with clear button
            col_info1, col_info2, col_info3 = st.columns([2, 1, 1])
            with col_info1:
                st.success(f"✅ Đang xem dataset hiện tại: {len(data)} dòng, {len(data.columns)} cột")
            with col_info2:
                st.info("💾 Dữ liệu đã lưu trong session")
            with col_info3:
                if st.button("🗑️ Xóa & Upload Mới", use_container_width=True, key="clear_and_upload"):
                    clear_data_related_state()
                    st.success("✅ Đã xóa! Upload file mới bên dưới.")
                    st.rerun()
            
            st.markdown("---")
            
            # Show FULL EDA tabs (same as when file is uploaded)
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Dữ Liệu Mẫu", 
                "📊 Thống Kê Mô Tả", 
                "📈 Phân Phối Dữ Liệu",
                "🤖 Phân Tích AI"
            ])
            
            # Copy the full tab content from the uploaded_file section
            # Tab 1: Sample Data
            with tab1:
                st.markdown("### 📋 Dữ Liệu Mẫu")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"📊 Hiển thị toàn bộ {len(data):,} dòng dữ liệu")
                with col2:
                    show_charts = st.checkbox("Hiện biểu đồ", value=True, key="show_charts_cached")
                
                display_data = data.copy()
                
                if show_charts:
                    st.markdown("---")
                    
                    import base64
                    from io import BytesIO
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    
                    header_html = "<div style='overflow-x: auto;'><table style='width: 100%; border-collapse: collapse; font-size: 0.85rem;'>"
                    header_html += "<tr style='background-color: #1e1e1e;'>"
                    
                    for col_name in data.columns:
                        col_data = data[col_name]
                        header_html += f"<td style='border: 1px solid #444; padding: 10px; text-align: center; vertical-align: top; min-width: 120px;'>"
                        header_html += f"<div style='font-weight: bold; margin-bottom: 5px;'>{col_name}</div>"
                        
                        if pd.api.types.is_numeric_dtype(col_data):
                            col_clean = col_data.dropna()
                            if len(col_clean) > 0:
                                fig, ax = plt.subplots(figsize=(1.5, 0.8), facecolor='none')
                                ax.hist(col_clean, bins=min(15, max(5, len(col_clean) // 10)), color='#667eea', edgecolor='none')
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                ax.spines['bottom'].set_visible(False)
                                ax.spines['left'].set_visible(False)
                                ax.patch.set_alpha(0)
                                
                                buffer = BytesIO()
                                plt.savefig(buffer, format='png', bbox_inches='tight', transparent=True, dpi=50)
                                buffer.seek(0)
                                img_base64 = base64.b64encode(buffer.read()).decode()
                                plt.close(fig)
                                
                                header_html += f"<img src='data:image/png;base64,{img_base64}' style='width: 100%; max-width: 120px;'/>"
                                header_html += f"<div style='font-size: 0.7rem; margin-top: 3px;'>Min: {col_clean.min():.1f} | Max: {col_clean.max():.1f}</div>"
                                header_html += f"<div style='font-size: 0.7rem;'>Mean: {col_clean.mean():.1f} | Unique: {col_data.nunique()}</div>"
                        else:
                            value_counts = col_data.value_counts().head(3)
                            total = len(col_data)
                            
                            if len(value_counts) > 0:
                                percentages = (value_counts / total * 100)
                                
                                fig, ax = plt.subplots(figsize=(1.5, 0.8), facecolor='none')
                                ax.barh(range(len(value_counts)), percentages.values, color='#764ba2')
                                ax.set_yticks(range(len(value_counts)))
                                ax.set_yticklabels([str(v)[:8] for v in value_counts.index], fontsize=6, color='white')
                                ax.set_xticks([])
                                ax.spines['top'].set_visible(False)
                                ax.spines['right'].set_visible(False)
                                ax.spines['bottom'].set_visible(False)
                                ax.spines['left'].set_visible(False)
                                ax.patch.set_alpha(0)
                                ax.invert_yaxis()
                                
                                for i, (idx, pct) in enumerate(zip(value_counts.index, percentages.values)):
                                    ax.text(pct + 2, i, f'{pct:.0f}%', va='center', fontsize=6, color='white')
                                
                                buffer = BytesIO()
                                plt.savefig(buffer, format='png', bbox_inches='tight', transparent=True, dpi=50)
                                buffer.seek(0)
                                img_base64 = base64.b64encode(buffer.read()).decode()
                                plt.close(fig)
                                
                                header_html += f"<img src='data:image/png;base64,{img_base64}' style='width: 100%; max-width: 120px;'/>"
                                header_html += f"<div style='font-size: 0.7rem; margin-top: 3px;'>Unique: {col_data.nunique()} | Mode: {str(value_counts.index[0])[:10]}</div>"
                        
                        missing_count = col_data.isnull().sum()
                        missing_pct = (missing_count / len(col_data) * 100) if len(col_data) > 0 else 0
                        if missing_count > 0:
                            header_html += f"<div style='font-size: 0.65rem; color: #ffaa00; margin-top: 2px;'>⚠️ Missing: {missing_count} ({missing_pct:.1f}%)</div>"
                        else:
                            header_html += f"<div style='font-size: 0.65rem; color: #44ff44; margin-top: 2px;'>✅ No missing</div>"
                        
                        header_html += "</td>"
                    
                    header_html += "</tr></table></div>"
                    st.markdown(header_html, unsafe_allow_html=True)
                
                st.markdown("---")
                st.dataframe(display_data, use_container_width=True, height=500)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Tổng số dòng", f"{len(data):,}")
                with col2:
                    st.metric("📋 Tổng số cột", len(data.columns))
                with col3:
                    missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
                    st.metric("❓ Dữ liệu thiếu", f"{missing_pct:.1f}%")
                with col4:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    st.metric("🔢 Cột số", len(numeric_cols))
            
            # Tab 2, 3, 4: Copy FULL content from uploaded section
            with tab2:
                st.markdown("### 📊 Thống Kê Mô Tả")
                
                # Numeric columns stats
                numeric_data = data.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    st.markdown("#### 🔢 Biến Số")
                    
                    stats_df = numeric_data.describe().T
                    stats_df['missing'] = data[numeric_data.columns].isnull().sum()
                    stats_df['missing_pct'] = (stats_df['missing'] / len(data) * 100).round(2)
                    
                    # Highlight styling
                    st.dataframe(
                        stats_df.style.background_gradient(cmap='viridis', subset=['mean', 'std']),
                        use_container_width=True
                    )
                    
                    # Download stats
                    csv = stats_df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        "📥 Tải Thống Kê (CSV)",
                        csv,
                        "statistics.csv",
                        "text/csv",
                        key='download-stats-cached'
                    )
                
                # Categorical columns
                categorical_data = data.select_dtypes(include=['object', 'category'])
                if not categorical_data.empty:
                    st.markdown("---")
                    st.markdown("#### � Biến Phân Loại")
                    
                    cat_info = []
                    for col in categorical_data.columns:
                        cat_info.append({
                            'Tên cột': col,
                            'Số giá trị khác nhau': data[col].nunique(),
                            'Giá trị phổ biến nhất': data[col].mode()[0] if not data[col].mode().empty else 'N/A',
                            'Tần suất cao nhất': data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0,
                            'Thiếu': data[col].isnull().sum(),
                            'Tỷ lệ thiếu (%)': f"{data[col].isnull().sum() / len(data) * 100:.2f}"
                        })
                    
                    cat_df = pd.DataFrame(cat_info)
                    st.dataframe(cat_df, use_container_width=True)
            
            with tab3:
                st.markdown("### 📈 Phân Phối & Tương Quan Dữ Liệu")
                
                viz_type = st.radio(
                    "Chọn loại phân tích:",
                    ["Correlation Heatmap", "Scatter Plot (2 Biến)"],
                    horizontal=True,
                    key="viz_type_cached"
                )
                
                if viz_type == "Correlation Heatmap":
                    st.markdown("#### 🔥 Ma Trận Tương Quan")
                    
                    numeric_data = data.select_dtypes(include=[np.number])
                    if not numeric_data.empty and len(numeric_data.columns) > 1:
                        corr_matrix = numeric_data.corr()
                        
                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            text_auto='.2f',
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="Ma trận tương quan giữa các biến",
                            zmin=-1,
                            zmax=1
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Find high correlations
                        st.markdown("#### 🔍 Các Cặp Biến Có Tương Quan Cao")
                        
                        threshold = st.slider("Ngưỡng tương quan:", 0.5, 0.95, 0.7, 0.05, key="cached_corr_threshold")
                        
                        high_corr = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                if abs(corr_matrix.iloc[i, j]) >= threshold:
                                    high_corr.append({
                                        'Biến 1': corr_matrix.columns[i],
                                        'Biến 2': corr_matrix.columns[j],
                                        'Tương quan': f"{corr_matrix.iloc[i, j]:.3f}",
                                        'Loại': 'Dương' if corr_matrix.iloc[i, j] > 0 else 'Âm'
                                    })
                        
                        if high_corr:
                            st.dataframe(pd.DataFrame(high_corr), use_container_width=True, hide_index=True)
                        else:
                            st.info(f"Không tìm thấy cặp biến nào có tương quan >= {threshold}")
                    else:
                        st.warning("Cần ít nhất 2 biến số để tạo ma trận tương quan.")
                
                else:  # Scatter Plot (2 Biến)
                    st.markdown("#### 📊 Phân Tích Chi Tiết 2 Biến")
                    
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            x_var = st.selectbox("Chọn biến X:", numeric_cols, key="cached_scatter_x")
                        with col2:
                            y_vars = [col for col in numeric_cols if col != x_var]
                            y_var = st.selectbox("Chọn biến Y:", y_vars, key="cached_scatter_y")
                        
                        # Options
                        opt_col1, opt_col2 = st.columns(2)
                        with opt_col1:
                            show_trendline = st.checkbox("Hiện đường xu hướng", value=True, key="cached_scatter_trend")
                        with opt_col2:
                            show_marginal = st.checkbox("Hiện phân phối biên", value=True, key="cached_scatter_marginal")
                        
                        # Create scatter plot
                        fig = px.scatter(
                            data,
                            x=x_var,
                            y=y_var,
                            trendline="ols" if show_trendline else None,
                            marginal_x="histogram" if show_marginal else None,
                            marginal_y="histogram" if show_marginal else None,
                            opacity=0.6,
                            title=f"Mối quan hệ giữa {x_var} và {y_var}"
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate correlation
                        corr = data[x_var].corr(data[y_var])
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("Tương quan Pearson", f"{corr:.3f}")
                        with metric_col2:
                            if abs(corr) >= 0.7:
                                st.metric("Mức độ", "Mạnh 💪", delta="Tương quan cao")
                            elif abs(corr) >= 0.4:
                                st.metric("Mức độ", "Trung bình ⚖️", delta="Tương quan vừa")
                            else:
                                st.metric("Mức độ", "Yếu �", delta="Tương quan thấp")
                        with metric_col3:
                            st.metric("Loại", "Dương ↗️" if corr > 0 else "Âm ↘️")
                    else:
                        st.warning("Cần ít nhất 2 biến số.")
            
            with tab4:
                st.markdown("### 🤖 Phân Tích Tự Động Bằng AI")
                
                # Check LLM configuration
                is_llm_configured = LLMConfig.is_configured()
                
                if not is_llm_configured:
                    st.info("""
                    ℹ️ **Chưa cấu hình LLM API**
                    
                    Để sử dụng phân tích AI chi tiết, vui lòng:
                    1. Tạo file `.env` trong thư mục gốc
                    2. Thêm Google API key: `GOOGLE_API_KEY=...`
                    3. (Tùy chọn) Chọn model: `GOOGLE_MODEL=gemini-2.5-flash`
                    4. (Tùy chọn) Chọn provider: `LLM_PROVIDER=google`
                    
                    **Lấy Google API key miễn phí tại: https://aistudio.google.com/app/apikey**
                    
                    **Hiện tại sẽ sử dụng chế độ phân tích tự động cơ bản.**
                    """)
                
                st.markdown("""
                <div style="background-color: #262730; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                    <h4 style="margin-top: 0; color: #667eea;">💡 Phân Tích Tự Động</h4>
                    <p>AI sẽ phân tích toàn bộ kết quả EDA và cung cấp:</p>
                    <ul>
                        <li>✨ Đánh giá chất lượng dữ liệu tổng thể</li>
                        <li>📊 Nhận xét về phân phối các biến quan trọng</li>
                        <li>🔗 Phát hiện tương quan và mối quan hệ giữa các biến</li>
                        <li>⚠️ Cảnh báo về outliers, missing data và vấn đề tiềm ẩn</li>
                        <li>💡 Đề xuất roadmap tiền xử lý dữ liệu chi tiết</li>
                        <li>🎯 Dự đoán khả năng xây dựng mô hình hiệu quả</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Options
                col1, col2 = st.columns([3, 1])
                with col1:
                    analysis_button = st.button(
                        "🔄 Tạo Phân Tích AI" if is_llm_configured else "📊 Tạo Phân Tích Tự Động",
                        use_container_width=True,
                        type="primary",
                        key="ai_analysis_btn_cached"
                    )
                with col2:
                    show_raw_summary = st.checkbox("Xem EDA Summary", value=False, key="show_eda_raw_cached")
                
                # Show raw EDA summary if requested
                if show_raw_summary:
                    st.markdown("---")
                    st.markdown("#### 📋 EDA Summary (Raw Data)")
                    with st.expander("Xem dữ liệu thống kê chi tiết", expanded=False):
                        summary_text = get_eda_summary(data, format="text")
                        st.text(summary_text)
                
                # Generate AI analysis
                if analysis_button:
                    with st.spinner("🤖 Đang phân tích dữ liệu..." if is_llm_configured else "📊 Đang tạo báo cáo..."):
                        try:
                            # Get API key and provider from config
                            api_key = LLMConfig.get_api_key() if is_llm_configured else None
                            provider = LLMConfig.DEFAULT_PROVIDER
                            
                            # Analyze with LLM
                            analysis_result = analyze_eda_with_llm(data, api_key=api_key, provider=provider)
                            
                            # Store in session state
                            st.session_state.ai_analysis = analysis_result
                            
                            st.success("✅ Phân tích hoàn thành!" if is_llm_configured else "✅ Báo cáo đã được tạo!")
                        
                        except Exception as e:
                            st.error(f"❌ Lỗi khi tạo phân tích: {str(e)}")
                            st.info("💡 Vui lòng kiểm tra API key và kết nối internet.")
                            import traceback
                            st.code(traceback.format_exc())
                
                # Display analysis if available
                if 'ai_analysis' in st.session_state and st.session_state.ai_analysis:
                    st.markdown("---")
                    st.markdown("### 📝 Kết Quả Phân Tích")
                    
                    # Display in a nice container
                    with st.container():
                        st.markdown(st.session_state.ai_analysis)
                    
                    # Auto-generate preprocessing suggestions after analysis
                    st.markdown("---")
                    st.markdown("### 💡 Gợi Ý Tiền Xử Lý")
                    
                    # Check if we already have suggestions
                    if 'preprocessing_suggestions' not in st.session_state:
                        # Auto-generate on first time
                        with st.spinner("🤖 Đang tạo gợi ý tiền xử lý từ AI..."):
                            try:
                                # Call LLM to generate preprocessing suggestions
                                provider = LLMConfig.DEFAULT_PROVIDER
                                api_key = LLMConfig.get_api_key() if is_llm_configured else None
                                
                                # Create prompt for preprocessing suggestions
                                suggestions_prompt = f"""Dựa trên kết quả phân tích EDA sau đây, hãy tạo một roadmap tiền xử lý dữ liệu theo ĐÚNG 4 BƯỚC sau:

KẾT QUẢ PHÂN TÍCH EDA:
{st.session_state.ai_analysis}

YÊU CẦU:
Trả về roadmap tiền xử lý theo ĐÚNG 4 BƯỚC SAU (không được thêm bớt bước):

**Bước 1: Xử Lý Biến Định Danh & Giá Trị Không Hợp Lệ**
- Xác định và loại bỏ các cột định danh (ID, customer_id, ...)
- Phát hiện và xử lý các giá trị không hợp lệ (âm, ngoài phạm vi, ...)
- Liệt kê CỤ THỂ tên cột cần xử lý

**Bước 2: Xử Lý Giá Trị Thiếu**
- Xác định các cột có missing values
- Đề xuất phương pháp xử lý CHO TỪNG CỘT (drop, imputation, ...)
- Giải thích lý do chọn phương pháp đó

**Bước 3: Xử Lý Outliers & Biến Đổi Phân Phối**
- Xác định các cột có outliers nghiêm trọng
- Đề xuất phương pháp xử lý outliers (Winsorization, IQR, Log transform, ...)
- Đề xuất biến đổi phân phối nếu cần (Log, Box-Cox, ...)

**Bước 4: Mã Hóa Biến Phân Loại**
- Xác định các biến phân loại cần mã hóa
- Đề xuất phương pháp mã hóa CHO TỪNG CỘT (One-Hot, Label Encoding, Target Encoding, ...)
- Giải thích lý do chọn phương pháp đó (dựa vào cardinality, mối quan hệ với target, ...)

FORMAT:
- Mỗi bước phải có tiêu đề in đậm với emoji
- Dưới mỗi bước là danh sách gạch đầu dòng CHI TIẾT, CỤ THỂ
- Đề cập TÊN CỘT và PHƯƠNG PHÁP cụ thể
- Ngôn ngữ: Tiếng Việt chuyên nghiệp

QUAN TRỌNG: 
- PHẢI trả về ĐÚNG 4 BƯỚC theo cấu trúc trên
- Mỗi bước phải CỤ THỂ, đề cập tên cột và phương pháp
- KHÔNG thêm bước khác, KHÔNG tóm tắt chung chung
- CHỈ trả về 4 bước, KHÔNG giải thích thêm!"""

                                # Call LLM
                                if is_llm_configured and api_key:
                                    if provider == "google":
                                        import google.generativeai as genai
                                        genai.configure(api_key=api_key)
                                        model = genai.GenerativeModel(LLMConfig.get_model(provider))
                                        response = model.generate_content(suggestions_prompt)
                                        suggestions_text = response.text.strip()
                                    else:
                                        # Fallback for other providers or no API
                                        suggestions_text = """**📋 Roadmap Tiền Xử Lý Dữ Liệu:**

**Bước 1: 🔍 Xử Lý Biến Định Danh & Giá Trị Không Hợp Lệ**
- Xác định và loại bỏ các cột định danh (customer_id, ID, ...)
- Kiểm tra và xử lý các giá trị không hợp lệ (âm, ngoài phạm vi hợp lý)

**Bước 2: ❓ Xử Lý Giá Trị Thiếu**
- Xác định các cột có missing values
- Áp dụng phương pháp phù hợp: Drop, Mean/Median Imputation, hoặc Forward Fill

**Bước 3: ⚠️ Xử Lý Outliers & Biến Đổi Phân Phối**
- Phát hiện outliers bằng phương pháp IQR hoặc Z-score
- Áp dụng Winsorization hoặc Log Transform cho các cột có outliers
- Biến đổi phân phối lệch bằng Log hoặc Box-Cox nếu cần

**Bước 4: 🔤 Mã Hóa Biến Phân Loại**
- One-Hot Encoding cho biến có cardinality thấp (< 10 categories)
- Label Encoding cho biến ordinal hoặc binary
- Target Encoding cho biến có cardinality cao"""
                                else:
                                    suggestions_text = """**📋 Roadmap Tiền Xử Lý Dữ Liệu:**

**Bước 1: 🔍 Xử Lý Biến Định Danh & Giá Trị Không Hợp Lệ**
- Xác định và loại bỏ các cột định danh (customer_id, ID, ...)
- Kiểm tra và xử lý các giá trị không hợp lệ (âm, ngoài phạm vi hợp lý)

**Bước 2: ❓ Xử Lý Giá Trị Thiếu**
- Xác định các cột có missing values
- Áp dụng phương pháp phù hợp: Drop, Mean/Median Imputation, hoặc Forward Fill

**Bước 3: ⚠️ Xử Lý Outliers & Biến Đổi Phân Phối**
- Phát hiện outliers bằng phương pháp IQR hoặc Z-score
- Áp dụng Winsorization hoặc Log Transform cho các cột có outliers
- Biến đổi phân phối lệch bằng Log hoặc Box-Cox nếu cần

**Bước 4: 🔤 Mã Hóa Biến Phân Loại**
- One-Hot Encoding cho biến có cardinality thấp (< 10 categories)
- Label Encoding cho biến ordinal hoặc binary
- Target Encoding cho biến có cardinality cao"""
                                
                                # Save to session state
                                st.session_state.preprocessing_suggestions = suggestions_text
                                st.session_state.eda_analysis_result = st.session_state.ai_analysis
                                st.session_state.llm_provider = provider
                                
                            except Exception as e:
                                st.session_state.preprocessing_suggestions = f"⚠️ Không thể tạo gợi ý tự động: {str(e)}\n\nVui lòng xem phân tích EDA ở trên để tự đưa ra các bước tiền xử lý."
                    
                    # Display suggestions
                    if 'preprocessing_suggestions' in st.session_state:
                        st.markdown(st.session_state.preprocessing_suggestions)
                        
                        # Button to regenerate
                        if st.button("🔄 Tạo Lại Gợi Ý", key="regenerate_preprocessing_suggestions_cached"):
                            del st.session_state.preprocessing_suggestions
                            st.rerun()
                    
                    # Download option
                    st.markdown("---")
                    st.download_button(
                        label="📥 Tải xuống phân tích (Markdown)",
                        data=st.session_state.ai_analysis,
                        file_name="eda_analysis.md",
                        mime="text/markdown",
                        use_container_width=True,
                        key="download_analysis_cached"
                    )
                else:
                    st.markdown("---")
                    st.info("👆 Nhấn nút phía trên để bắt đầu phân tích!")
            
            return
        
        # No data at all - show sample format
        print("DEBUG: No file uploaded, showing sample format")
        st.info("📝 No file uploaded. Please select a CSV file.")
        
        with st.expander("📋 View Sample Format"):
            st.markdown("""
            CSV file should have the following format:
            
            | customer_id | age | income | credit_history | loan_amount | ... | default |
            |-------------|-----|--------|----------------|-------------|-----|---------|
            | 1001        | 35  | 50000  | good           | 10000       | ... | 0       |
            | 1002        | 42  | 75000  | excellent      | 15000       | ... | 0       |
            | 1003        | 28  | 30000  | poor           | 5000        | ... | 1       |
            
            - Last column is target: 0 = no default, 1 = default
            - Other columns are features
            """)
            
            # Simple basic sample only - no complex loading
            print("DEBUG: Creating basic sample data")
            sample_data = pd.DataFrame({
                'customer_id': range(1001, 1011),
                'age': np.random.randint(25, 65, 10),
                'income': np.random.randint(30000, 100000, 10),
                'credit_history': np.random.choice(['good', 'fair', 'poor'], 10),
                'loan_amount': np.random.randint(5000, 50000, 10),
                'default': np.random.choice([0, 1], 10, p=[0.8, 0.2])
            })
            
            print(f"DEBUG: Sample data created: {len(sample_data)} rows")
            
            csv = sample_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Sample Data",
                csv,
                "sample_credit_data.csv",
                "text/csv"
            )
            
            st.dataframe(sample_data, use_container_width=True)

