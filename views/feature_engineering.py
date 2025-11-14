"""
Trang Xử Lý & Chọn Biến - Feature Engineering & Selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.ui_components import show_processing_placeholder
from utils.session_state import init_session_state

def render():
    """Render trang xử lý và chọn biến"""
    init_session_state()
    
    st.markdown("## ⚙️ Xử Lý & Chọn Biến")
    st.markdown("Tiền xử lý dữ liệu và lựa chọn các đặc trưng quan trọng cho mô hình.")
    
    # Check if data exists
    if st.session_state.data is None:
        st.warning("⚠️ Chưa có dữ liệu. Vui lòng upload dữ liệu từ trang 'Upload & EDA' trước.")
        return
    
    # Initialize backup system - save original data on first visit
    if 'data_original_backup' not in st.session_state:
        st.session_state.data_original_backup = st.session_state.data.copy()
        st.session_state.column_backups = {}  # Store backup before each column processing
    
    data = st.session_state.data
    
    # Show data selector if processed data exists
    if st.session_state.get('processed_data') is not None:
        col_selector1, col_selector2 = st.columns([3, 1])
        with col_selector1:
            st.success(f"✅ Đang làm việc với dataset: {len(data)} dòng, {len(data.columns)} cột")
        with col_selector2:
            data_view = st.selectbox(
                "Xem dữ liệu:",
                ["Original", "Processed"],
                key="data_view_selector",
                help="Chọn xem dữ liệu gốc hoặc đã xử lý"
            )
            if data_view == "Processed":
                data = st.session_state.processed_data
                st.info(f"📊 Processed: {len(data)} dòng")
    else:
        st.success(f"✅ Đang làm việc với dataset: {len(data)} dòng, {len(data.columns)} cột")
    
    # Add clear configuration button
    col_status1, col_status2, col_status3 = st.columns([2, 1, 1])
    with col_status2:
        # Show number of configurations
        total_configs = (
            len(st.session_state.get('missing_config', {})) +
            len(st.session_state.get('encoding_config', {})) +
            len(st.session_state.get('binning_config', {}))
        )
        if total_configs > 0:
            st.info(f"📋 {total_configs} cấu hình đã lưu")
    
    with col_status3:
        if total_configs > 0:
            if st.button("� Hoàn Về Ban Đầu", key="clear_all_configs", help="Xóa tất cả cấu hình và hoàn về dữ liệu gốc", type="primary"):
                # Restore original data
                st.session_state.data = st.session_state.data_original_backup.copy()
                # Clear all configs
                st.session_state.missing_config = {}
                st.session_state.encoding_config = {}
                st.session_state.scaling_config = {}
                st.session_state.outlier_config = {}
                st.session_state.binning_config = {}
                # Clear column backups
                st.session_state.column_backups = {}
                st.success("✅ Đã hoàn về dữ liệu ban đầu!")
                st.rerun()
    
    st.markdown("---")
    
    # Tabs for different processing steps
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Tiền Xử Lý",
        "📊 Binning",
        "⭐ Feature Importance",
        "✅ Chọn Biến"
    ])
    
    # Tab 1: Preprocessing
    with tab1:
        st.markdown("### 🔧 Các Bước Tiền Xử Lý")
        
        # Show saved configurations summary at the top
        if st.session_state.get('missing_config') or st.session_state.get('processed_data') is not None:
            st.markdown("#### 📌 Trạng Thái Xử Lý")
            
            status_col1, status_col2, status_col3 = st.columns(3)
            
            with status_col1:
                if st.session_state.get('processed_data') is not None:
                    st.success(f"✅ Đã xử lý: {len(st.session_state.processed_data)} dòng")
                else:
                    st.info("○ Chưa áp dụng xử lý")
            
            with status_col2:
                missing_configs = len(st.session_state.get('missing_config', {}))
                if missing_configs > 0:
                    st.info(f"📝 {missing_configs} cấu hình Missing")
                else:
                    st.caption("Chưa có cấu hình")
            
            with status_col3:
                if st.session_state.get('processed_data') is not None:
                    original_missing = data.isnull().sum().sum()
                    processed_missing = st.session_state.processed_data.isnull().sum().sum()
                    reduced = original_missing - processed_missing
                    st.metric("Đã giảm missing", f"{reduced}", delta=f"-{reduced}")
            
            st.markdown("---")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 1️⃣ Xử Lý Giá Trị Thiếu")
            
            missing_data = data.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                st.warning(f"⚠️ Có {len(missing_data)} cột chứa giá trị thiếu")
                
                # Display missing data summary
                missing_df = pd.DataFrame({
                    'Cột': missing_data.index,
                    'Số lượng thiếu': missing_data.values,
                    'Tỷ lệ (%)': (missing_data.values / len(data) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
            
            else:
                st.success("✅ Không có giá trị thiếu trong dataset")
            
            # Show missing patterns if data has missing
            missing_data_temp = data.isnull().sum()
            missing_data_temp = missing_data_temp[missing_data_temp > 0]
            
            if len(missing_data_temp) > 0:
                st.markdown("##### 📈 Phân Tích Mẫu Thiếu")
                
                # Calculate missing percentage by column
                missing_pct_chart = (missing_data_temp / len(data) * 100).sort_values(ascending=False)
                
                # Create simple bar chart
                import plotly.express as px
                fig = px.bar(
                    x=missing_pct_chart.values,
                    y=missing_pct_chart.index,
                    orientation='h',
                    labels={'x': 'Tỷ lệ (%)', 'y': 'Cột'},
                    title="Tỷ lệ dữ liệu thiếu theo cột"
                )
                fig.update_layout(
                    template="plotly_dark",
                    height=300,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                fig.update_traces(marker_color='#ff6b6b')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✨ Dữ liệu hoàn chỉnh, không có giá trị thiếu!")
        
        with col2:
            st.markdown("##### 📊 Gợi Ý & Thống Kê")
            
            suggestions = st.session_state.get("preprocessing_suggestions")
            if suggestions:
                st.markdown("""
                <div style="background-color: #262730; padding: 1.2rem; border-radius: 10px; border-left: 4px solid #667eea;">
                    <h4 style="margin-top: 0; color: #667eea; font-size: 1.1rem;">💡 Gợi Ý Xử Lý (AI)</h4>
                """, unsafe_allow_html=True)
                st.markdown(suggestions)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                # Show default processing tips
                st.markdown("""
                <div style="background-color: #262730; padding: 1.2rem; border-radius: 10px; border-left: 4px solid #667eea;">
                    <h4 style="margin-top: 0; color: #667eea; font-size: 1.1rem;">💡 Gợi Ý Xử Lý</h4>
                    <ul style="font-size: 0.9rem; margin-bottom: 0;">
                        <li><strong>Mean</strong>: Tốt cho dữ liệu phân phối chuẩn</li>
                        <li><strong>Median</strong>: Tốt khi có outliers</li>
                        <li><strong>Mode</strong>: Cho biến phân loại</li>
                        <li><strong>Forward/Backward Fill</strong>: Cho time series</li>
                        <li><strong>Interpolation</strong>: Cho dữ liệu liên tục</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Show rows with missing data section (moved outside columns)
        if len(missing_data) > 0:
            st.markdown("---")
            st.markdown("##### 📋 Xem Bản Ghi Có Dữ Liệu Thiếu")
            
            # Get rows with any missing values
            rows_with_missing = data[data.isnull().any(axis=1)]
            
            col_preview1, col_preview2 = st.columns([3, 2])
            with col_preview1:
                st.metric("Số dòng có missing", len(rows_with_missing), 
                         f"{len(rows_with_missing)/len(data)*100:.1f}% tổng số")
            with col_preview2:
                show_missing_rows = st.checkbox("Hiển thị các dòng", value=True, key="show_missing_rows")
            
            if show_missing_rows:
                # Filter options - select column to prioritize
                selected_col_filter = st.selectbox(
                    "Ưu tiên hiển thị cột thiếu:",
                    ["Tất cả"] + list(missing_data.index),
                    key="missing_col_filter",
                    help="Chọn cột để ưu tiên sắp xếp các dòng thiếu dữ liệu ở cột đó lên trên. Tất cả các dòng sẽ được hiển thị."
                )
                
                # Sort data to prioritize rows with missing data in selected column
                if selected_col_filter != "Tất cả":
                    # Create a priority column: 1 if selected column is missing, 0 otherwise
                    rows_display = rows_with_missing.copy()
                    rows_display['_priority'] = rows_display[selected_col_filter].isnull().astype(int)
                    # Sort by priority (missing in selected column first), then by index
                    rows_display = rows_display.sort_values('_priority', ascending=False)
                    # Drop priority column - SHOW ALL ROWS
                    display_data = rows_display.drop('_priority', axis=1)
                    
                    # Show info about filtering
                    missing_in_selected = rows_with_missing[selected_col_filter].isnull().sum()
                    st.info(f"🎯 Ưu tiên: {missing_in_selected} dòng thiếu dữ liệu ở `{selected_col_filter}` được sắp xếp lên trên. Hiển thị tất cả {len(display_data):,} dòng.")
                else:
                    # SHOW ALL rows with missing data
                    display_data = rows_with_missing
                
                # Highlight missing values with special color for selected column
                def highlight_missing(val):
                    return 'background-color: #ff6b6b; color: white;' if pd.isnull(val) else ''
                
                def highlight_selected_col_missing(row):
                    # Special highlight for selected column if missing
                    styles = [''] * len(row)
                    for idx, (col_name, val) in enumerate(row.items()):
                        if pd.isnull(val):
                            if selected_col_filter != "Tất cả" and col_name == selected_col_filter:
                                # Brighter red for selected column
                                styles[idx] = 'background-color: #ff3333; color: white; font-weight: bold; border: 2px solid #ff0000;'
                            else:
                                # Normal red for other missing values
                                styles[idx] = 'background-color: #ff6b6b; color: white;'
                    return styles
                
                st.dataframe(
                    display_data.style.apply(highlight_selected_col_missing, axis=1),
                    use_container_width=True,
                    height=400
                )
            
            st.markdown("---")
            st.markdown("##### ⚙️ Cấu Hình Xử Lý Từng Cột")
            
            # Select column to configure
            selected_missing_col = st.selectbox(
                "Chọn cột để xử lý:",
                missing_data.index.tolist(),
                key="selected_missing_col"
            )
            
            # Show column info - simplified without nested columns
            col_type = data[selected_missing_col].dtype
            missing_count = missing_data[selected_missing_col]
            missing_pct = (missing_count / len(data) * 100)
            
            st.markdown(f"""
            **Kiểu dữ liệu:** `{col_type}` | **Số missing:** `{missing_count}` | **Tỷ lệ:** `{missing_pct:.1f}%`
            """)
            
            # Method selection based on data type
            if pd.api.types.is_numeric_dtype(data[selected_missing_col]):
                method_options = [
                    "Mean Imputation",
                    "Median Imputation",
                    "Mode Imputation",
                    "Forward Fill",
                    "Backward Fill",
                    "Interpolation",
                    "Constant Value",
                    "Drop Rows"
                ]
            else:
                method_options = [
                    "Mode Imputation",
                    "Forward Fill",
                    "Backward Fill",
                    "Constant Value",
                    "Drop Rows"
                ]
            
            selected_method = st.selectbox(
                "Phương pháp xử lý:",
                method_options,
                key=f"method_{selected_missing_col}"
            )
            
            # Constant value input if needed
            constant_val = None
            if selected_method == "Constant Value":
                constant_val = st.text_input(
                    "Giá trị:",
                    value="0" if pd.api.types.is_numeric_dtype(data[selected_missing_col]) else "Unknown",
                    key=f"const_{selected_missing_col}"
                )
            
            # Initialize session state for missing config
            if 'missing_config' not in st.session_state:
                st.session_state.missing_config = {}
            
            # Process button
            if st.button("✅ Xử Lý Ngay", key=f"add_config_{selected_missing_col}", use_container_width=True, type="primary"):
                with st.spinner(f"Đang xử lý cột `{selected_missing_col}`..."):
                    # BACKUP current state before processing
                    st.session_state.column_backups[selected_missing_col] = {
                        'data': st.session_state.data[selected_missing_col].copy(),
                        'full_data': st.session_state.data.copy()
                    }
                    
                    # Apply the method immediately to session data
                    if selected_method == "Mean Imputation":
                        st.session_state.data[selected_missing_col].fillna(
                            st.session_state.data[selected_missing_col].mean(), inplace=True)
                    elif selected_method == "Median Imputation":
                        st.session_state.data[selected_missing_col].fillna(
                            st.session_state.data[selected_missing_col].median(), inplace=True)
                    elif selected_method == "Mode Imputation":
                        mode_val = st.session_state.data[selected_missing_col].mode()
                        fill_val = mode_val[0] if len(mode_val) > 0 else 0
                        st.session_state.data[selected_missing_col].fillna(fill_val, inplace=True)
                    elif selected_method == "Forward Fill":
                        st.session_state.data[selected_missing_col].fillna(method='ffill', inplace=True)
                    elif selected_method == "Backward Fill":
                        st.session_state.data[selected_missing_col].fillna(method='bfill', inplace=True)
                    elif selected_method == "Interpolation":
                        st.session_state.data[selected_missing_col] = st.session_state.data[selected_missing_col].interpolate()
                    elif selected_method == "Constant Value":
                        fill_val = constant_val
                        if pd.api.types.is_numeric_dtype(st.session_state.data[selected_missing_col]):
                            fill_val = float(fill_val) if '.' in str(fill_val) else int(fill_val)
                        st.session_state.data[selected_missing_col].fillna(fill_val, inplace=True)
                    elif selected_method == "Drop Rows":
                        st.session_state.data = st.session_state.data[st.session_state.data[selected_missing_col].notna()]
                    
                    # Save to config history for tracking
                    st.session_state.missing_config[selected_missing_col] = {
                        'method': selected_method,
                        'original_missing': missing_count,
                        'processed': True,
                        'can_undo': True
                    }
                    if selected_method == "Constant Value":
                        st.session_state.missing_config[selected_missing_col]['constant'] = constant_val
                    
                    st.success(f"✅ Đã xử lý cột `{selected_missing_col}` bằng {selected_method}")
                    st.rerun()  # Refresh to update the display
            
            # Undo button
            if selected_missing_col in st.session_state.missing_config:
                if st.button("🔄 Hoàn Tác", key=f"remove_config_{selected_missing_col}", use_container_width=True):
                    # Restore from backup
                    if selected_missing_col in st.session_state.column_backups:
                        backup = st.session_state.column_backups[selected_missing_col]
                        
                        # Check if it was "Drop Rows" - need full data restore
                        config = st.session_state.missing_config[selected_missing_col]
                        if config['method'] == "Drop Rows":
                            st.session_state.data = backup['full_data'].copy()
                        else:
                            st.session_state.data[selected_missing_col] = backup['data'].copy()
                        
                        # Remove from config and backup
                        del st.session_state.missing_config[selected_missing_col]
                        del st.session_state.column_backups[selected_missing_col]
                        
                        st.success(f"✅ Đã hoàn tác xử lý cho cột `{selected_missing_col}`")
                        st.rerun()
                    else:
                        st.error("⚠️ Không tìm thấy backup cho cột này")
                        del st.session_state.missing_config[selected_missing_col]
                        st.rerun()
                
                
                # Show current configuration (Processing History)
                if st.session_state.missing_config:
                    st.markdown("---")
                    st.markdown("##### � Lịch Sử Xử Lý")
                    
                    config_df = pd.DataFrame([
                        {
                            'Cột': col,
                            'Phương pháp': cfg['method'],
                            'Missing ban đầu': f"{cfg['original_missing']}",
                            'Giá trị điền': cfg.get('constant', '-'),
                            'Trạng thái': '✅ Đã xử lý'
                        }
                        for col, cfg in st.session_state.missing_config.items()
                    ])
                    
                    st.dataframe(config_df, use_container_width=True, hide_index=True)
                    
                    # Clear all history button
                    if st.button("🗑️ Xóa Toàn Bộ Lịch Sử", key="clear_history", use_container_width=True):
                        st.session_state.missing_config = {}
                        st.success("✅ Đã xóa lịch sử xử lý")
                        st.rerun()
                else:
                    st.info("💡 Chưa xử lý cột nào. Chọn cột và phương pháp ở trên, sau đó bấm 'Xử Lý Ngay'.")
        
        # Section 2: Mã Hóa Biến Phân Loại (Moved to separate section)
        st.markdown("---")
        st.markdown("### 2️⃣ Mã Hóa Biến Phân Loại")
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            st.warning(f"⚠️ Có {len(categorical_cols)} biến phân loại cần mã hóa")
            
            # Show categorical columns summary
            col_enc1, col_enc2 = st.columns([1, 1])
            
            with col_enc1:
                st.markdown("##### 📋 Danh Sách Biến Phân Loại")
                
                cat_summary = []
                for col in categorical_cols:
                    unique_vals = data[col].nunique()
                    cat_summary.append({
                        'Cột': col,
                        'Số giá trị khác nhau': unique_vals,
                        'Giá trị phổ biến': data[col].mode()[0] if not data[col].mode().empty else 'N/A'
                    })
                
                cat_df = pd.DataFrame(cat_summary)
                st.dataframe(cat_df, use_container_width=True, hide_index=True)
            
            with col_enc2:
                st.markdown("##### ⚙️ Cấu Hình Mã Hóa Từng Cột")
                
                # Select column to encode
                selected_enc_col = st.selectbox(
                    "Chọn cột để mã hóa:",
                    categorical_cols,
                    key="selected_enc_col"
                )
                
                # Show column info
                unique_count = data[selected_enc_col].nunique()
                st.metric("Số giá trị khác nhau", unique_count)
                
                # Encoding method selection
                encoding_method = st.selectbox(
                    "Phương pháp mã hóa:",
                    ["One-Hot Encoding", "Label Encoding", "Target Encoding", "Ordinal Encoding"],
                    key="encoding_method"
                )
                
                # Initialize encoding config
                if 'encoding_config' not in st.session_state:
                    st.session_state.encoding_config = {}
                
                # Add configuration button
                enc_btn_col1, enc_btn_col2 = st.columns(2)
                with enc_btn_col1:
                    if st.button("➕ Thêm Cấu Hình", key="add_enc_config", use_container_width=True):
                        st.session_state.encoding_config[selected_enc_col] = {
                            'method': encoding_method,
                            'unique_count': unique_count
                        }
                        st.success(f"✅ Đã thêm cấu hình cho {selected_enc_col}")
                
                with enc_btn_col2:
                    if selected_enc_col in st.session_state.encoding_config:
                        if st.button("�️ Xóa", key="remove_enc_config", use_container_width=True):
                            del st.session_state.encoding_config[selected_enc_col]
                            st.success(f"✅ Đã xóa cấu hình")
                            st.rerun()
            
            # Show current encoding configurations
            if st.session_state.encoding_config:
                st.markdown("---")
                st.markdown("##### 📝 Cấu Hình Mã Hóa Hiện Tại")
                
                enc_config_df = pd.DataFrame([
                    {
                        'Cột': col,
                        'Phương pháp': cfg['method'],
                        'Số giá trị': cfg['unique_count']
                    }
                    for col, cfg in st.session_state.encoding_config.items()
                ])
                
                st.dataframe(enc_config_df, use_container_width=True, hide_index=True)
                
                # Apply all encoding configurations
                if st.button("✅ Áp Dụng Tất Cả Mã Hóa", type="primary", use_container_width=True, key="apply_all_encoding"):
                    with st.spinner("Đang mã hóa..."):
                        show_processing_placeholder(f"Mã hóa {len(st.session_state.encoding_config)} biến phân loại")
                        st.success(f"✅ Đã mã hóa {len(st.session_state.encoding_config)} biến!")
                        st.info("💡 Cấu hình mã hóa đã được lưu lại.")
            else:
                st.info("💡 Chưa có cấu hình mã hóa nào. Hãy chọn cột và phương pháp ở trên.")
        
        else:
            st.success("✅ Không có biến phân loại cần mã hóa")
        
        # Section 3: Chuẩn Hóa/Scale
        st.markdown("---")
        st.markdown("### 3️⃣ Chuẩn Hóa/Scaling")
        
        col_scale1, col_scale2 = st.columns([1, 1])
        
        with col_scale1:
            st.markdown("##### ⚙️ Cấu Hình Scaling")
            
            scaling_method = st.selectbox(
                "Phương pháp:",
                ["Standard Scaler", "Min-Max Scaler", "Robust Scaler", "No Scaling"],
                key="scaling_method",
                help="Standard Scaler: z-score normalization\nMin-Max: scale to [0,1]\nRobust: resistant to outliers"
            )
            
            # Select columns to scale
            numeric_cols_for_scale = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols_for_scale:
                selected_scale_cols = st.multiselect(
                    "Chọn các cột cần scaling:",
                    numeric_cols_for_scale,
                    default=numeric_cols_for_scale[:min(5, len(numeric_cols_for_scale))],
                    key="selected_scale_cols"
                )
                
                if st.button("✅ Áp Dụng Scaling", key="apply_scaling", use_container_width=True, type="primary"):
                    if selected_scale_cols:
                        with st.spinner(f"Đang scaling {len(selected_scale_cols)} cột..."):
                            show_processing_placeholder(f"Scaling {len(selected_scale_cols)} cột với {scaling_method}")
                            st.success(f"✅ Đã scaling {len(selected_scale_cols)} cột!")
                    else:
                        st.warning("Vui lòng chọn ít nhất 1 cột để scaling")
            else:
                st.info("Không có cột số nào để scaling")
        
        with col_scale2:
            st.markdown("##### 📊 Thông Tin")
            
            st.info("""
            **Standard Scaler**: Chuẩn hóa về mean=0, std=1
            
            **Min-Max Scaler**: Scale về khoảng [0, 1]
            
            **Robust Scaler**: Sử dụng median và IQR, tốt cho data có outliers
            """)
        
        # Section 4: Xử Lý Outliers
        st.markdown("---")
        st.markdown("### 4️⃣ Xử Lý Outliers")
        
        col_outlier1, col_outlier2 = st.columns([1, 1])
        
        with col_outlier1:
            st.markdown("##### ⚙️ Cấu Hình Xử Lý Outliers")
            
            outlier_method = st.selectbox(
                "Phương pháp:",
                ["IQR Method", "Z-Score", "Winsorization", "Keep All"],
                key="outlier_method",
                help="IQR: Sử dụng Interquartile Range\nZ-Score: Dựa trên độ lệch chuẩn\nWinsorization: Thay thế outliers bằng giá trị ngưỡng"
            )
            
            if outlier_method != "Keep All":
                threshold = st.slider(
                    "Ngưỡng:",
                    1.0, 5.0, 1.5 if outlier_method == "IQR Method" else 3.0, 0.5,
                    key="outlier_threshold"
                )
            
            numeric_cols_for_outlier = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols_for_outlier:
                selected_outlier_cols = st.multiselect(
                    "Chọn các cột cần xử lý outliers:",
                    numeric_cols_for_outlier,
                    key="selected_outlier_cols"
                )
                
                if st.button("✅ Xử Lý Outliers", key="apply_outliers", use_container_width=True, type="primary"):
                    if selected_outlier_cols:
                        with st.spinner(f"Đang xử lý outliers..."):
                            show_processing_placeholder(f"Xử lý outliers bằng {outlier_method}")
                            st.success(f"✅ Đã xử lý outliers cho {len(selected_outlier_cols)} cột!")
                    else:
                        st.warning("Vui lòng chọn ít nhất 1 cột")
        
        with col_outlier2:
            st.markdown("##### 📊 Thống Kê Outliers")
            
            if numeric_cols_for_outlier:
                # Show outlier statistics
                outlier_stats = []
                for col in numeric_cols_for_outlier[:5]:  # Show first 5
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                        outlier_pct = len(outliers) / len(col_data) * 100
                        
                        outlier_stats.append({
                            'Cột': col,
                            'Số outliers': len(outliers),
                            'Tỷ lệ (%)': f"{outlier_pct:.2f}"
                        })
                
                if outlier_stats:
                    st.dataframe(pd.DataFrame(outlier_stats), use_container_width=True, hide_index=True)
        
        # Section 5: Cân Bằng Dữ Liệu
        st.markdown("---")
        st.markdown("### 5️⃣ Cân Bằng Dữ Liệu")
        
        col_balance1, col_balance2 = st.columns([1, 1])
        
        with col_balance1:
            st.markdown("##### ⚙️ Cấu Hình Balancing")
            
            balance_method = st.selectbox(
                "Phương pháp:",
                ["SMOTE", "Random Over-sampling", "Random Under-sampling", "No Balancing"],
                key="balance_method",
                help="SMOTE: Synthetic Minority Over-sampling\nOver-sampling: Nhân bản class thiểu số\nUnder-sampling: Giảm class đa số"
            )
            
            if st.button("✅ Cân Bằng Dữ Liệu", key="apply_balance", use_container_width=True, type="primary"):
                with st.spinner("Đang cân bằng dữ liệu..."):
                    show_processing_placeholder(f"Cân bằng dữ liệu bằng {balance_method}")
                    st.success("✅ Đã cân bằng dữ liệu!")
        
        with col_balance2:
            st.markdown("##### 📊 Phân Bổ Class")
            
            # Try to detect target column
            potential_targets = [col for col in data.columns if 'target' in col.lower() or 'default' in col.lower() or 'label' in col.lower()]
            
            if potential_targets:
                target_col = potential_targets[0]
                class_dist = data[target_col].value_counts()
                
                st.metric("Target column", target_col)
                for cls, count in class_dist.items():
                    st.text(f"Class {cls}: {count} ({count/len(data)*100:.1f}%)")
            else:
                st.info("Chưa xác định được target column. Vui lòng chọn target ở tab 'Chọn Biến'.")
    
    # Tab 2: Binning
    with tab2:
        st.markdown("### 📊 Phân Nhóm (Binning) Biến Liên Tục")
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p style="margin: 0;">💡 <strong>Binning</strong> giúp chuyển biến liên tục thành các nhóm rời rạc, 
            hữu ích cho việc phân tích và giải thích mô hình.</p>
        </div>
        """, unsafe_allow_html=True)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_col = st.selectbox("Chọn biến để binning:", numeric_cols, key="binning_col")
                
                binning_method = st.radio(
                    "Phương pháp binning:",
                    ["Equal Width", "Equal Frequency", "Custom"],
                    key="binning_method"
                )
                
                num_bins = st.slider("Số nhóm:", 2, 10, 5, key="num_bins")
                
                if st.button("🔄 Thực Hiện Binning", key="do_binning", type="primary"):
                    show_processing_placeholder(f"Binning biến {selected_col} thành {num_bins} nhóm")
                    st.success(f"✅ Đã tạo biến mới: {selected_col}_binned")
            
            with col2:
                # Visualize binning
                st.markdown("#### 📊 Trực Quan Hóa Binning")
                
                # Create sample bins for visualization
                col_data = data[selected_col].dropna()
                
                # Mock binning visualization
                fig = go.Figure()
                
                # Histogram
                fig.add_trace(go.Histogram(
                    x=col_data,
                    nbinsx=num_bins,
                    name='Distribution',
                    marker_color='#667eea',
                    opacity=0.7
                ))
                
                # Add bin edges as vertical lines (mock)
                bin_edges = np.linspace(col_data.min(), col_data.max(), num_bins + 1)
                for edge in bin_edges:
                    fig.add_vline(x=edge, line_dash="dash", line_color="red", opacity=0.5)
                
                fig.update_layout(
                    title=f"Binning visualization - {selected_col}",
                    xaxis_title=selected_col,
                    yaxis_title="Frequency",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Bin statistics
                st.markdown("#### 📊 Thống Kê Từng Nhóm")
                bin_stats = pd.DataFrame({
                    'Nhóm': [f"Bin {i+1}" for i in range(num_bins)],
                    'Khoảng': [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(num_bins)],
                    'Số mẫu': np.random.randint(50, 200, num_bins),  # Mock data
                })
                st.dataframe(bin_stats, use_container_width=True)
        else:
            st.warning("⚠️ Không có biến số nào trong dataset")
    
    # Tab 3: Feature Importance
    with tab3:
        st.markdown("### ⭐ Mức Độ Quan Trọng Của Đặc Trưng")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("#### ⚙️ Cấu Hình")
            
            importance_method = st.selectbox(
                "Phương pháp tính:",
                ["Random Forest", "LightGBM", "XGBoost", "Logistic Regression (Coef)"],
                key="importance_method"
            )
            
            top_n = st.slider("Top N features:", 5, 30, 15, key="top_n_features")
            
            if st.button("🔄 Tính Feature Importance", key="calc_importance", type="primary"):
                with st.spinner("Đang tính toán..."):
                    show_processing_placeholder(f"Tính feature importance bằng {importance_method}")
                    st.success("✅ Đã tính xong!")
        
        with col2:
            st.markdown("#### 📊 Biểu Đồ Feature Importance")
            
            # Mock feature importance data
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                sample_features = numeric_cols[:min(top_n, len(numeric_cols))]
                importance_scores = np.random.random(len(sample_features))
                importance_scores = importance_scores / importance_scores.sum()  # Normalize
                
                # Sort by importance
                sorted_indices = np.argsort(importance_scores)[::-1]
                sorted_features = [sample_features[i] for i in sorted_indices]
                sorted_scores = importance_scores[sorted_indices]
                
                # Create bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=sorted_scores,
                    y=sorted_features,
                    orientation='h',
                    marker=dict(
                        color=sorted_scores,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    ),
                    text=[f"{score:.3f}" for score in sorted_scores],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title=f"Top {len(sorted_features)} Important Features",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    template="plotly_dark",
                    height=max(400, len(sorted_features) * 25),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Lưu ý**: Đây là dữ liệu mô phỏng. Backend sẽ tính toán importance thực tế từ mô hình.")
            else:
                st.warning("⚠️ Không có biến số để tính feature importance")
    
    # Tab 4: Feature Selection
    with tab4:
        st.markdown("### ✅ Chọn Đặc Trưng Cho Mô Hình")
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p style="margin: 0;">📋 <strong>Chọn các đặc trưng</strong> bạn muốn sử dụng để huấn luyện mô hình. 
            Có thể dựa trên feature importance hoặc kiến thức nghiệp vụ.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get all columns except target
        all_cols = data.columns.tolist()
        
        # Assume last column is target (or let user select)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            target_col = st.selectbox(
                "Chọn biến mục tiêu (Target):",
                all_cols,
                index=len(all_cols) - 1 if len(all_cols) > 0 else 0,
                key="target_col"
            )
        
        with col2:
            st.metric("Số biến có sẵn", len(all_cols) - 1)
        
        # Available features (exclude target)
        available_features = [col for col in all_cols if col != target_col]
        
        # Feature selection
        st.markdown("#### 🎯 Chọn Đặc Trưng")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selection_mode = st.radio(
                "Chế độ chọn:",
                ["Chọn thủ công", "Chọn tự động (theo threshold)"],
                key="selection_mode"
            )
            
            if selection_mode == "Chọn tự động (theo threshold)":
                importance_threshold = st.slider(
                    "Ngưỡng importance:",
                    0.0, 1.0, 0.01, 0.01,
                    key="importance_threshold"
                )
                
                if st.button("🔄 Chọn Tự Động", key="auto_select"):
                    # Mock auto selection
                    num_selected = np.random.randint(5, min(15, len(available_features)))
                    selected = np.random.choice(available_features, num_selected, replace=False).tolist()
                    st.session_state.selected_features = selected
                    st.success(f"✅ Đã chọn tự động {len(selected)} đặc trưng!")
        
        with col2:
            # Manual selection
            if selection_mode == "Chọn thủ công":
                selected_features = st.multiselect(
                    "Chọn các đặc trưng:",
                    available_features,
                    default=st.session_state.selected_features if st.session_state.selected_features else available_features[:min(10, len(available_features))],
                    key="manual_features"
                )
                
                if st.button("💾 Lưu Lựa Chọn", key="save_selection", type="primary"):
                    st.session_state.selected_features = selected_features
                    st.success(f"✅ Đã lưu {len(selected_features)} đặc trưng!")
            else:
                # Display auto-selected features
                if st.session_state.selected_features:
                    st.multiselect(
                        "Đặc trưng đã chọn:",
                        available_features,
                        default=st.session_state.selected_features,
                        disabled=True,
                        key="auto_features_display"
                    )
        
        st.markdown("---")
        
        # Summary
        if st.session_state.selected_features:
            st.success(f"✅ **Đã chọn {len(st.session_state.selected_features)} đặc trưng cho mô hình**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                numeric_selected = len([f for f in st.session_state.selected_features 
                                       if f in data.select_dtypes(include=[np.number]).columns])
                st.metric("Biến số", numeric_selected)
            
            with col2:
                categorical_selected = len([f for f in st.session_state.selected_features 
                                           if f in data.select_dtypes(include=['object', 'category']).columns])
                st.metric("Biến phân loại", categorical_selected)
            
            with col3:
                st.metric("Tổng biến", len(st.session_state.selected_features))
            
            # Display selected features
            with st.expander("📋 Xem Danh Sách Đặc Trưng Đã Chọn"):
                for i, feat in enumerate(st.session_state.selected_features, 1):
                    st.text(f"{i}. {feat}")
        else:
            st.warning("⚠️ Chưa chọn đặc trưng nào. Vui lòng chọn ít nhất một đặc trưng.")

