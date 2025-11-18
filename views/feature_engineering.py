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
                st.session_state.removed_columns_config = {}
                st.session_state.missing_config = {}
                st.session_state.encoding_config = {}
                st.session_state.scaling_config = {}
                st.session_state.outlier_config = {}
                st.session_state.binning_config = {}
                st.session_state.validation_config = {}
                # Clear column backups
                st.session_state.column_backups = {}
                st.session_state.removed_columns_backup = {}
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
        
        # ============ DASHBOARD TỔNG HỢP CẤU HÌNH ============
        st.markdown("#### 📊 Dashboard Theo Dõi Cấu Hình")
        
        # Count all configurations
        total_configs = (
            len(st.session_state.get('removed_columns_config', {})) +
            len(st.session_state.get('missing_config', {})) +
            len(st.session_state.get('outlier_config', {}).get('columns', [])) +
            len(st.session_state.get('encoding_config', {})) +
            len(st.session_state.get('validation_config', {}))
        )
        
        if total_configs > 0:
            # Summary cards
            status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)
            
            with status_col1:
                removed_cols = len(st.session_state.get('removed_columns_config', {}))
                if removed_cols > 0:
                    st.metric("🗑️ Loại Bỏ Cột", removed_cols, delta="cột")
                else:
                    st.metric("🗑️ Loại Bỏ Cột", "0", delta="chưa có")
            
            with status_col2:
                missing_configs = len(st.session_state.get('missing_config', {}))
                if missing_configs > 0:
                    st.metric("📝 Missing Values", missing_configs, delta="cột")
                else:
                    st.metric("📝 Missing Values", "0", delta="chưa có")
            
            with status_col3:
                outlier_configs = len(st.session_state.get('outlier_config', {}).get('columns', []))
                if outlier_configs > 0:
                    st.metric("⚠️ Outliers", outlier_configs, delta="cột")
                else:
                    st.metric("⚠️ Outliers", "0", delta="chưa có")
            
            with status_col4:
                encoding_configs = len(st.session_state.get('encoding_config', {}))
                if encoding_configs > 0:
                    st.metric("🔤 Encoding", encoding_configs, delta="cột")
                else:
                    st.metric("🔤 Encoding", "0", delta="chưa có")
            
            with status_col5:
                validation_configs = len(st.session_state.get('validation_config', {}))
                if validation_configs > 0:
                    st.metric("✅ Validation", validation_configs, delta="cột")
                else:
                    st.metric("✅ Validation", "0", delta="chưa có")
            
            # Detailed configuration table
            st.markdown("##### 📋 Chi Tiết Cấu Hình Đã Lưu")
            
            # Create a container for configurations with undo buttons
            config_count = 0
            
            # Removed columns
            for col, cfg in st.session_state.get('removed_columns_config', {}).items():
                config_count += 1
                col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1.5, 2, 1.5, 1.5, 0.8])
                
                with col1:
                    st.markdown(f"**2️⃣ Loại Bỏ Cột**")
                with col2:
                    st.markdown(f"`{col}`")
                with col3:
                    st.markdown(f"{cfg.get('reason', 'Loại bỏ')}")
                with col4:
                    st.markdown(f"unique={cfg.get('unique_count', 'N/A')}")
                with col5:
                    st.markdown("✅ **Đã áp dụng**")
                with col6:
                    if st.button("↩️", key=f"undo_removed_{col}", help=f"Hoàn tác loại bỏ cột {col}"):
                        # Restore column from backup
                        if col in st.session_state.get('removed_columns_backup', {}):
                            st.session_state.data[col] = st.session_state.removed_columns_backup[col]
                            del st.session_state.removed_columns_backup[col]
                            del st.session_state.removed_columns_config[col]
                            st.success(f"✅ Đã khôi phục cột `{col}`")
                            st.rerun()
                
                st.markdown("---")
            
            # Missing configs
            for col, cfg in st.session_state.get('missing_config', {}).items():
                config_count += 1
                col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1.5, 2, 1.5, 1.5, 0.8])
                
                with col1:
                    st.markdown(f"**2️⃣ Missing Values**")
                with col2:
                    st.markdown(f"`{col}`")
                with col3:
                    st.markdown(f"{cfg.get('method', 'N/A')}")
                with col4:
                    st.markdown(f"{cfg.get('value', 'N/A')}")
                with col5:
                    st.markdown("⏳ **Chờ áp dụng**")
                with col6:
                    if st.button("🗑️", key=f"delete_missing_{col}", help=f"Xóa cấu hình {col}"):
                        del st.session_state.missing_config[col]
                        st.success(f"✅ Đã xóa cấu hình")
                        st.rerun()
                
                st.markdown("---")
            
            # Outlier configs
            if st.session_state.get('outlier_config'):
                outlier_cfg = st.session_state.outlier_config
                is_applied = 'info' in outlier_cfg
                
                for col in outlier_cfg.get('columns', []):
                    config_count += 1
                    col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1.5, 2, 1.5, 1.5, 0.8])
                    
                    with col1:
                        st.markdown(f"**4️⃣ Outliers**")
                    with col2:
                        st.markdown(f"`{col}`")
                    with col3:
                        st.markdown(f"{outlier_cfg.get('method', 'N/A')}")
                    with col4:
                        st.markdown(f"{outlier_cfg.get('multiplier', outlier_cfg.get('threshold', 'N/A'))}")
                    with col5:
                        if is_applied:
                            st.markdown("✅ **Đã áp dụng**")
                        else:
                            st.markdown("⏳ **Chờ áp dụng**")
                    with col6:
                        if is_applied and col in st.session_state.get('column_backups', {}):
                            if st.button("↩️", key=f"undo_outlier_{col}", help=f"Hoàn tác xử lý outlier {col}"):
                                # Restore column from backup
                                st.session_state.data[col] = st.session_state.column_backups[col]
                                del st.session_state.column_backups[col]
                                # Remove from outlier config
                                st.session_state.outlier_config['columns'].remove(col)
                                if not st.session_state.outlier_config['columns']:
                                    st.session_state.outlier_config = {}
                                st.success(f"✅ Đã hoàn tác xử lý outlier cho `{col}`")
                                st.rerun()
                    
                    st.markdown("---")
            
            # Encoding configs
            for col, cfg in st.session_state.get('encoding_config', {}).items():
                config_count += 1
                col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1.5, 2, 1.5, 1.5, 0.8])
                
                params_str = ''
                if 'params' in cfg:
                    params = cfg['params']
                    if 'drop_first' in params:
                        params_str = f"drop_first={params['drop_first']}"
                    elif 'target_column' in params:
                        params_str = f"target={params['target_column']}"
                
                is_applied = cfg.get('applied', False)
                
                with col1:
                    st.markdown(f"**5️⃣ Encoding**")
                with col2:
                    st.markdown(f"`{col}`")
                with col3:
                    st.markdown(f"{cfg.get('method', 'N/A')}")
                with col4:
                    st.markdown(f"{params_str or 'default'}")
                with col5:
                    if is_applied:
                        st.markdown("✅ **Đã áp dụng**")
                    else:
                        st.markdown("⏳ **Chờ áp dụng**")
                with col6:
                    if is_applied and f"encoding_{col}" in st.session_state.get('column_backups', {}):
                        if st.button("↩️", key=f"undo_encoding_{col}", help=f"Hoàn tác mã hóa {col}"):
                            # Restore original column
                            backup_key = f"encoding_{col}"
                            st.session_state.data[col] = st.session_state.column_backups[backup_key]
                            del st.session_state.column_backups[backup_key]
                            
                            # Remove encoded columns if One-Hot
                            if col in st.session_state.get('encoding_applied_info', {}):
                                enc_info = st.session_state.encoding_applied_info[col]
                                if 'new_columns' in enc_info:
                                    for new_col in enc_info['new_columns']:
                                        if new_col in st.session_state.data.columns:
                                            st.session_state.data.drop(columns=[new_col], inplace=True)
                                del st.session_state.encoding_applied_info[col]
                            
                            # Remove from encoding config
                            del st.session_state.encoding_config[col]
                            st.success(f"✅ Đã hoàn tác mã hóa cho `{col}`")
                            st.rerun()
                    elif not is_applied:
                        if st.button("🗑️", key=f"delete_encoding_{col}", help=f"Xóa cấu hình {col}"):
                            del st.session_state.encoding_config[col]
                            st.success(f"✅ Đã xóa cấu hình")
                            st.rerun()
                
                st.markdown("---")
            
            # Validation configs
            for col, cfg in st.session_state.get('validation_config', {}).items():
                config_count += 1
                col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1.5, 2, 1.5, 1.5, 0.8])
                
                is_applied = cfg.get('applied', False)
                
                with col1:
                    st.markdown(f"**2️⃣ Validation**")
                with col2:
                    st.markdown(f"`{col}`")
                with col3:
                    st.markdown(f"{cfg.get('type', 'N/A')}")
                with col4:
                    st.markdown(f"{cfg.get('threshold', cfg.get('value', 'N/A'))}")
                with col5:
                    if is_applied:
                        st.markdown("✅ **Đã áp dụng**")
                    else:
                        st.markdown("⏳ **Chờ áp dụng**")
                with col6:
                    if is_applied and f"validation_{col}" in st.session_state.get('column_backups', {}):
                        if st.button("↩️", key=f"undo_validation_{col}", help=f"Hoàn tác validation {col}"):
                            # Restore column from backup
                            backup_key = f"validation_{col}"
                            st.session_state.data[col] = st.session_state.column_backups[backup_key]
                            del st.session_state.column_backups[backup_key]
                            del st.session_state.validation_config[col]
                            st.success(f"✅ Đã hoàn tác validation cho `{col}`")
                            st.rerun()
                    elif not is_applied:
                        if st.button("🗑️", key=f"delete_validation_{col}", help=f"Xóa cấu hình {col}"):
                            del st.session_state.validation_config[col]
                            st.success(f"✅ Đã xóa cấu hình")
                            st.rerun()
                
                st.markdown("---")
            
            if config_count == 0:
                st.info("💡 Chưa có cấu hình nào. Hãy thêm cấu hình ở các bước bên dưới.")
            
            # Action buttons
            st.markdown("---")
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("🔄 Làm Mới Dashboard", key="refresh_dashboard", use_container_width=True):
                    st.rerun()
            
            with action_col2:
                # Export configuration as JSON
                if st.button("📥 Xuất Cấu Hình", key="export_config", use_container_width=True):
                    import json
                    config_export = {
                        'removed_columns': st.session_state.get('removed_columns_config', {}),
                        'validation': st.session_state.get('validation_config', {}),
                        'missing': st.session_state.get('missing_config', {}),
                        'outlier': st.session_state.get('outlier_config', {}),
                        'encoding': st.session_state.get('encoding_config', {})
                    }
                    config_json = json.dumps(config_export, indent=2, default=str)
                    st.download_button(
                        "💾 Tải JSON",
                        config_json,
                        "preprocessing_config.json",
                        "application/json",
                        key="download_config_json"
                    )
            
            with action_col3:
                # Clear all pending configs
                pending_missing = len([c for c in st.session_state.get('missing_config', {}).items() if not c[1].get('applied')])
                pending_encoding = len([c for c in st.session_state.get('encoding_config', {}).items() if not c[1].get('applied')])
                pending_validation = len([c for c in st.session_state.get('validation_config', {}).items() if not c[1].get('applied')])
                pending_count = pending_missing + pending_encoding + pending_validation
                
                if pending_count > 0:
                    if st.button(f"🗑️ Xóa {pending_count} Chờ Áp Dụng", key="clear_pending", use_container_width=True, type="secondary"):
                        # Clear only pending configs
                        # Keep applied encoding configs
                        st.session_state.encoding_config = {
                            col: cfg for col, cfg in st.session_state.get('encoding_config', {}).items()
                            if cfg.get('applied', False)
                        }
                        # Keep applied validation configs
                        st.session_state.validation_config = {
                            col: cfg for col, cfg in st.session_state.get('validation_config', {}).items()
                            if cfg.get('applied', False)
                        }
                        # Clear pending missing configs
                        st.session_state.missing_config = {}
                        st.success(f"✅ Đã xóa {pending_count} cấu hình chờ áp dụng!")
                        st.rerun()
            
            # Summary statistics
            st.markdown("---")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                pending = pending_count if 'pending_count' in locals() else 0
                st.info(f"⏳ **{pending}** cấu hình chờ áp dụng")
            
            with summary_col2:
                applied = (
                    len(st.session_state.get('removed_columns_config', {})) +
                    len([c for c in st.session_state.get('encoding_config', {}).items() if c[1].get('applied')]) +
                    len([c for c in st.session_state.get('validation_config', {}).items() if c[1].get('applied')]) +
                    (len(st.session_state.get('outlier_config', {}).get('columns', [])) if 'info' in st.session_state.get('outlier_config', {}) else 0)
                )
                st.success(f"✅ **{applied}** cấu hình đã áp dụng")
            
            with summary_col3:
                total = applied + pending
                st.metric("📊 Tổng cộng", f"{total} cấu hình")
            
            st.markdown("---")
        else:
            st.info("💡 Chưa có cấu hình nào được lưu. Hãy bắt đầu thêm cấu hình ở các bước bên dưới!")
            st.markdown("---")
        
        # ============ END DASHBOARD ============
        
        st.markdown("### 1️⃣ Tổng Quan Dữ Liệu Thiếu")
        
        # Calculate missing data
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
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
        
        # Section 2: Xử Lý Biến Định Danh & Giá Trị Không Hợp Lệ
        st.markdown("### 2️⃣ Xử Lý Biến Định Danh & Giá Trị Không Hợp Lệ")
        
        col_id1, col_id2 = st.columns([1, 1])
        
        with col_id1:
            st.markdown("##### 🔍 Xóa/Loại Biến Định Danh")
            
            st.markdown("""
            <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <p style="margin: 0; font-size: 0.9rem;">💡 <strong>Biến định danh</strong> không mang thông tin dự đoán, 
                nên loại bỏ khỏi mô hình:</p>
                <ul style="font-size: 0.85rem; margin: 0.5rem 0 0 1rem;">
                    <li>ID khách hàng (customer_id, user_id)</li>
                    <li>Số hợp đồng (contract_id, loan_id)</li>
                    <li>Số CMND/CCCD, số tài khoản</li>
                    <li>Các mã định danh khác</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all columns with unique count info
            all_cols = data.columns.tolist()
            
            if all_cols:
                st.info(f"📋 Dataset hiện có {len(all_cols)} cột")
                
                # Show columns info
                cols_info = []
                for col in all_cols:
                    unique_count = data[col].nunique()
                    unique_pct = round(unique_count / len(data) * 100, 2)
                    cols_info.append({
                        'Cột': col,
                        'Số giá trị duy nhất': unique_count,
                        'Tỷ lệ unique (%)': unique_pct
                    })
                
                cols_df = pd.DataFrame(cols_info)
                st.dataframe(cols_df, use_container_width=True, hide_index=True, height=300)
                
                # Select columns to remove
                cols_to_remove = st.multiselect(
                    "Chọn cột để loại bỏ:",
                    all_cols,
                    key="id_cols_to_remove",
                    help="Chọn các cột định danh cần loại bỏ khỏi dataset"
                )
                
                if st.button("🗑️ Loại Bỏ Các Cột Đã Chọn", key="remove_id_cols", use_container_width=True, type="primary"):
                    if cols_to_remove:
                        # Initialize removed_columns_config if not exists
                        if 'removed_columns_config' not in st.session_state:
                            st.session_state.removed_columns_config = {}
                        
                        # Backup before removing
                        if 'removed_columns_backup' not in st.session_state:
                            st.session_state.removed_columns_backup = {}
                        
                        for col in cols_to_remove:
                            # Backup data
                            st.session_state.removed_columns_backup[col] = st.session_state.data[col].copy()
                            
                            # Save to config for dashboard tracking
                            st.session_state.removed_columns_config[col] = {
                                'reason': 'Biến định danh',
                                'unique_count': st.session_state.data[col].nunique(),
                                'applied': True
                            }
                            
                            # Remove from data
                            st.session_state.data = st.session_state.data.drop(columns=[col])
                        
                        st.success(f"✅ Đã loại bỏ {len(cols_to_remove)} cột!")
                        st.rerun()
                    else:
                        st.warning("Vui lòng chọn ít nhất 1 cột")
        
        with col_id2:
            st.markdown("##### ⚠️ Xử Lý Giá Trị Không Hợp Lệ")
            
            st.markdown("""
            <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <p style="margin: 0; font-size: 0.9rem;">💡 <strong>Ví dụ giá trị vô lý cần xử lý:</strong></p>
                <ul style="font-size: 0.85rem; margin: 0.5rem 0 0 1rem;">
                    <li>Thu nhập âm → 0 hoặc NA</li>
                    <li>Tuổi < 18 hoặc > 90 → ngưỡng</li>
                    <li>Dư nợ âm → 0</li>
                    <li>Kỳ hạn ≤ 0 → NA hoặc min</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Select column to validate
            numeric_cols_validate = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols_validate:
                selected_validate_col = st.selectbox(
                    "Chọn cột cần xử lý:",
                    numeric_cols_validate,
                    key="validate_col",
                    help="Chọn cột số để kiểm tra và xử lý giá trị không hợp lệ"
                )
                
                # Show statistics
                col_data_valid = data[selected_validate_col].dropna()
                if len(col_data_valid) > 0:
                    col_min = col_data_valid.min()
                    col_max = col_data_valid.max()
                    col_mean = col_data_valid.mean()
                    
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("Min", f"{col_min:.2f}")
                    with stat_col2:
                        st.metric("Mean", f"{col_mean:.2f}")
                    with stat_col3:
                        st.metric("Max", f"{col_max:.2f}")
                    
                    st.markdown("---")
                    
                    # Configure validation rule
                    validation_type = st.selectbox(
                        "Loại quy tắc:",
                        ["Giá trị âm", "Ngưỡng tối thiểu", "Ngưỡng tối đa", "Khoảng giá trị"],
                        key="validation_type"
                    )
                    
                    if validation_type == "Giá trị âm":
                        invalid_count = len(data[data[selected_validate_col] < 0])
                        st.info(f"📊 Tìm thấy **{invalid_count}** giá trị âm")
                        
                        action = st.radio(
                            "Hành động:",
                            ["Chuyển về 0", "Chuyển về NA"],
                            key="negative_action"
                        )
                        
                        # Initialize validation config
                        if 'validation_config' not in st.session_state:
                            st.session_state.validation_config = {}
                        
                        if st.button("✅ Áp Dụng", key="apply_negative", use_container_width=True, type="primary"):
                            if invalid_count > 0:
                                # Backup before applying
                                if 'column_backups' not in st.session_state:
                                    st.session_state.column_backups = {}
                                backup_key = f"validation_{selected_validate_col}"
                                st.session_state.column_backups[backup_key] = st.session_state.data[selected_validate_col].copy()
                                
                                # Save config
                                st.session_state.validation_config[selected_validate_col] = {
                                    'type': validation_type,
                                    'action': action,
                                    'affected_count': invalid_count,
                                    'applied': True
                                }
                                
                                # Apply
                                if action == "Chuyển về 0":
                                    st.session_state.data.loc[st.session_state.data[selected_validate_col] < 0, selected_validate_col] = 0
                                else:
                                    st.session_state.data.loc[st.session_state.data[selected_validate_col] < 0, selected_validate_col] = np.nan
                                st.success(f"✅ Đã xử lý {invalid_count} giá trị âm!")
                                st.rerun()
                            else:
                                st.info("Không có giá trị âm để xử lý")
                    
                    elif validation_type == "Ngưỡng tối thiểu":
                        min_threshold = st.number_input(
                            "Ngưỡng min (giá trị < ngưỡng sẽ bị xử lý):",
                            value=float(col_min),
                            key="min_threshold"
                        )
                        invalid_count = len(data[data[selected_validate_col] < min_threshold])
                        st.info(f"📊 Tìm thấy **{invalid_count}** giá trị < {min_threshold}")
                        
                        action = st.radio(
                            "Hành động:",
                            [f"Chuyển về {min_threshold}", "Chuyển về NA"],
                            key="min_action"
                        )
                        
                        if st.button("✅ Áp Dụng", key="apply_min", use_container_width=True, type="primary"):
                            if invalid_count > 0:
                                # Backup before applying
                                if 'column_backups' not in st.session_state:
                                    st.session_state.column_backups = {}
                                backup_key = f"validation_{selected_validate_col}"
                                st.session_state.column_backups[backup_key] = st.session_state.data[selected_validate_col].copy()
                                
                                # Save config
                                if 'validation_config' not in st.session_state:
                                    st.session_state.validation_config = {}
                                
                                st.session_state.validation_config[selected_validate_col] = {
                                    'type': validation_type,
                                    'threshold': min_threshold,
                                    'action': action,
                                    'affected_count': invalid_count,
                                    'applied': True
                                }
                                
                                # Apply
                                if "NA" in action:
                                    st.session_state.data.loc[st.session_state.data[selected_validate_col] < min_threshold, selected_validate_col] = np.nan
                                else:
                                    st.session_state.data.loc[st.session_state.data[selected_validate_col] < min_threshold, selected_validate_col] = min_threshold
                                st.success(f"✅ Đã xử lý {invalid_count} giá trị!")
                                st.rerun()
                    
                    elif validation_type == "Ngưỡng tối đa":
                        max_threshold = st.number_input(
                            "Ngưỡng max (giá trị > ngưỡng sẽ bị xử lý):",
                            value=float(col_max),
                            key="max_threshold"
                        )
                        invalid_count = len(data[data[selected_validate_col] > max_threshold])
                        st.info(f"📊 Tìm thấy **{invalid_count}** giá trị > {max_threshold}")
                        
                        action = st.radio(
                            "Hành động:",
                            [f"Chuyển về {max_threshold}", "Chuyển về NA"],
                            key="max_action"
                        )
                        
                        if st.button("✅ Áp Dụng", key="apply_max", use_container_width=True, type="primary"):
                            if invalid_count > 0:
                                # Backup before applying
                                if 'column_backups' not in st.session_state:
                                    st.session_state.column_backups = {}
                                backup_key = f"validation_{selected_validate_col}"
                                st.session_state.column_backups[backup_key] = st.session_state.data[selected_validate_col].copy()
                                
                                # Save config
                                if 'validation_config' not in st.session_state:
                                    st.session_state.validation_config = {}
                                
                                st.session_state.validation_config[selected_validate_col] = {
                                    'type': validation_type,
                                    'threshold': max_threshold,
                                    'action': action,
                                    'affected_count': invalid_count,
                                    'applied': True
                                }
                                
                                # Apply
                                if "NA" in action:
                                    st.session_state.data.loc[st.session_state.data[selected_validate_col] > max_threshold, selected_validate_col] = np.nan
                                else:
                                    st.session_state.data.loc[st.session_state.data[selected_validate_col] > max_threshold, selected_validate_col] = max_threshold
                                st.success(f"✅ Đã xử lý {invalid_count} giá trị!")
                                st.rerun()
                    
                    elif validation_type == "Khoảng giá trị":
                        col_range1, col_range2 = st.columns(2)
                        with col_range1:
                            range_min = st.number_input("Min:", value=float(col_min), key="range_min")
                        with col_range2:
                            range_max = st.number_input("Max:", value=float(col_max), key="range_max")
                        
                        invalid_count = len(data[(data[selected_validate_col] < range_min) | (data[selected_validate_col] > range_max)])
                        st.info(f"📊 Tìm thấy **{invalid_count}** giá trị ngoài [{range_min}, {range_max}]")
                        
                        action = st.radio(
                            "Hành động:",
                            ["Clamp về ngưỡng", "Chuyển về NA"],
                            key="range_action",
                            help="Clamp: giới hạn giá trị trong khoảng min-max"
                        )
                        
                        if st.button("✅ Áp Dụng", key="apply_range", use_container_width=True, type="primary"):
                            if invalid_count > 0:
                                # Backup before applying
                                if 'column_backups' not in st.session_state:
                                    st.session_state.column_backups = {}
                                backup_key = f"validation_{selected_validate_col}"
                                st.session_state.column_backups[backup_key] = st.session_state.data[selected_validate_col].copy()
                                
                                # Save config
                                if 'validation_config' not in st.session_state:
                                    st.session_state.validation_config = {}
                                
                                st.session_state.validation_config[selected_validate_col] = {
                                    'type': validation_type,
                                    'range': f'[{range_min}, {range_max}]',
                                    'action': action,
                                    'affected_count': invalid_count,
                                    'applied': True
                                }
                                
                                # Apply
                                if action == "Clamp về ngưỡng":
                                    st.session_state.data[selected_validate_col] = st.session_state.data[selected_validate_col].clip(range_min, range_max)
                                else:
                                    mask = (st.session_state.data[selected_validate_col] < range_min) | (st.session_state.data[selected_validate_col] > range_max)
                                    st.session_state.data.loc[mask, selected_validate_col] = np.nan
                                st.success(f"✅ Đã xử lý {invalid_count} giá trị!")
                                st.rerun()
                else:
                    st.warning("Cột này không có dữ liệu hợp lệ")
            else:
                st.info("Không có cột số nào để kiểm tra")
        
        st.markdown("---")
        
        # Section 3: Xử Lý Giá Trị Thiếu
        st.markdown("### 3️⃣ Xử Lý Giá Trị Thiếu")
        
        # Show rows with missing data section (moved outside columns)
        if len(missing_data) > 0:
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
        
        # Section 4: Xử Lý Outliers & Biến Đổi Phân Phối
        st.markdown("---")
        st.markdown("### 4️⃣ Xử Lý Outliers & Biến Đổi Phân Phối")
        
        # Sub-section 4.1: Xử Lý Outliers
        st.markdown("#### 4.1 Xử Lý Outliers")
        
        col_outlier1, col_outlier2 = st.columns([1, 1])
        
        with col_outlier1:
            st.markdown("##### ⚙️ Cấu Hình Xử Lý Outliers")
            
            outlier_method = st.selectbox(
                "Phương pháp:",
                ["Winsorization", "IQR Method", "Z-Score", "Keep All"],
                key="outlier_method",
                help="Winsorization: Thay outliers bằng phân vị\nIQR: Sử dụng Interquartile Range\nZ-Score: Dựa trên độ lệch chuẩn\nKeep All: Giữ nguyên"
            )
            
            # Show method-specific parameters
            if outlier_method == "Winsorization":
                st.markdown("**Cấu hình phân vị:**")
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    lower_percentile = st.number_input(
                        "Phân vị dưới:",
                        min_value=0.0,
                        max_value=0.5,
                        value=0.05,
                        step=0.01,
                        key="winsor_lower",
                        help="Ví dụ: 0.05 = 5% (thay outliers dưới 5% bằng giá trị 5%)"
                    )
                with col_w2:
                    upper_percentile = st.number_input(
                        "Phân vị trên:",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.95,
                        step=0.01,
                        key="winsor_upper",
                        help="Ví dụ: 0.95 = 95% (thay outliers trên 95% bằng giá trị 95%)"
                    )
            
            elif outlier_method == "IQR Method":
                st.markdown("**Cấu hình IQR:**")
                col_iqr1, col_iqr2 = st.columns(2)
                with col_iqr1:
                    iqr_multiplier = st.slider(
                        "Hệ số IQR:",
                        min_value=1.0,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        key="iqr_multiplier",
                        help="Ngưỡng = Q1 - k*IQR và Q3 + k*IQR"
                    )
                with col_iqr2:
                    iqr_action = st.selectbox(
                        "Hành động:",
                        ["clip", "remove", "nan"],
                        key="iqr_action",
                        help="clip: cắt về ngưỡng\nremove: xóa dòng\nnan: thay bằng NaN"
                    )
            
            elif outlier_method == "Z-Score":
                st.markdown("**Cấu hình Z-Score:**")
                col_z1, col_z2 = st.columns(2)
                with col_z1:
                    z_threshold = st.slider(
                        "Ngưỡng Z-score:",
                        min_value=2.0,
                        max_value=4.0,
                        value=3.0,
                        step=0.1,
                        key="z_threshold",
                        help="Giá trị có |z-score| > ngưỡng sẽ được xử lý"
                    )
                with col_z2:
                    z_action = st.selectbox(
                        "Hành động:",
                        ["clip", "remove", "nan"],
                        key="z_action",
                        help="clip: cắt về ngưỡng\nremove: xóa dòng\nnan: thay bằng NaN"
                    )
            
            numeric_cols_for_outlier = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols_for_outlier:
                selected_outlier_cols = st.multiselect(
                    "Chọn các cột cần xử lý outliers:",
                    numeric_cols_for_outlier,
                    key="selected_outlier_cols",
                    help="Chọn các cột số cần phát hiện và xử lý outliers"
                )
                
                if st.button("✅ Xử Lý Outliers", key="apply_outliers", use_container_width=True, type="primary"):
                    if selected_outlier_cols:
                        with st.spinner(f"Đang xử lý outliers bằng {outlier_method}..."):
                            try:
                                # Import backend handler
                                from backend.data_processing import handle_outliers
                                
                                # Backup columns before processing
                                if 'column_backups' not in st.session_state:
                                    st.session_state.column_backups = {}
                                
                                for col in selected_outlier_cols:
                                    st.session_state.column_backups[col] = st.session_state.data[col].copy()
                                
                                # Prepare parameters based on method
                                kwargs = {}
                                if outlier_method == "Winsorization":
                                    kwargs = {
                                        'lower_percentile': lower_percentile,
                                        'upper_percentile': upper_percentile
                                    }
                                elif outlier_method == "IQR Method":
                                    kwargs = {
                                        'multiplier': iqr_multiplier,
                                        'action': iqr_action
                                    }
                                elif outlier_method == "Z-Score":
                                    kwargs = {
                                        'threshold': z_threshold,
                                        'action': z_action
                                    }
                                
                                # Apply outlier handling
                                processed_data, outlier_info = handle_outliers(
                                    data=st.session_state.data,
                                    method=outlier_method,
                                    columns=selected_outlier_cols,
                                    **kwargs
                                )
                                
                                # Save to session state
                                st.session_state.data = processed_data
                                st.session_state.outlier_config = {
                                    'method': outlier_method,
                                    'columns': selected_outlier_cols,
                                    'info': outlier_info,
                                    **kwargs
                                }
                                
                                st.success(f"✅ Đã xử lý outliers cho {len(selected_outlier_cols)} cột bằng {outlier_method}!")
                                
                                # Show summary
                                total_outliers = sum(info.get('outliers_count', info.get('outliers_detected', 0)) 
                                                   for info in outlier_info.values())
                                st.info(f"📊 Tổng số outliers đã xử lý: **{total_outliers}**")
                                
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Lỗi khi xử lý outliers: {str(e)}")
                                import traceback
                                with st.expander("Chi tiết lỗi"):
                                    st.code(traceback.format_exc())
                    else:
                        st.warning("Vui lòng chọn ít nhất 1 cột")
        
        with col_outlier2:
            st.markdown("##### 📊 Thống Kê Outliers")
            
            # Show saved outlier config if exists
            if st.session_state.get('outlier_config'):
                config = st.session_state.outlier_config
                
                st.markdown(f"""
                <div style="background-color: #1a472a; padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981; margin-bottom: 1rem;">
                    <p style="margin: 0; font-weight: bold; color: #10b981;">✅ Đã Xử Lý</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                        <strong>Phương pháp:</strong> {config['method']}<br>
                        <strong>Số cột:</strong> {len(config['columns'])}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show detailed info for each column
                if 'info' in config:
                    outlier_summary = []
                    for col, info in config['info'].items():
                        outliers_count = info.get('outliers_count', info.get('outliers_detected', 0))
                        outliers_pct = info.get('outliers_percentage', 0)
                        
                        outlier_summary.append({
                            'Cột': col,
                            'Outliers': outliers_count,
                            'Tỷ lệ (%)': f"{outliers_pct:.2f}",
                            'Phương pháp': info.get('method', config['method'])
                        })
                    
                    if outlier_summary:
                        st.dataframe(
                            pd.DataFrame(outlier_summary),
                            use_container_width=True,
                            hide_index=True,
                            height=min(300, len(outlier_summary) * 40 + 50)
                        )
                        
                        # Show detailed report in expander
                        with st.expander("📋 Xem Báo Cáo Chi Tiết"):
                            for col, info in config['info'].items():
                                st.markdown(f"**{col}**")
                                
                                info_items = []
                                for key, value in info.items():
                                    if key not in ['method', 'outliers_mask']:
                                        if isinstance(value, (int, float)):
                                            if isinstance(value, float):
                                                info_items.append(f"- {key}: {value:.4f}")
                                            else:
                                                info_items.append(f"- {key}: {value}")
                                        else:
                                            info_items.append(f"- {key}: {value}")
                                
                                st.markdown("\n".join(info_items))
                                st.markdown("---")
            
            elif numeric_cols_for_outlier:
                st.info("⚙️ Cấu hình và áp dụng xử lý outliers ở bên trái")
                
                # Show outlier detection for preview
                st.markdown("**Preview (Top 5 cột):**")
                outlier_stats = []
                for col in numeric_cols_for_outlier[:5]:
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
                        outlier_pct = len(outliers) / len(col_data) * 100
                        
                        outlier_stats.append({
                            'Cột': col,
                            'Outliers': len(outliers),
                            'Tỷ lệ (%)': f"{outlier_pct:.2f}"
                        })
                
                if outlier_stats:
                    st.dataframe(
                        pd.DataFrame(outlier_stats),
                        use_container_width=True,
                        hide_index=True
                    )
                    st.caption("💡 Sử dụng phương pháp IQR (k=1.5) để preview")
            else:
                st.info("Không có cột số nào để phân tích outliers")
        
        # Sub-section 4.2: Biến Đổi Phân Phối
        st.markdown("---")
        st.markdown("#### 4.2 Biến Đổi Phân Phối")
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <p style="margin: 0; font-size: 0.9rem;">💡 <strong>Biến đổi phân phối</strong> giúp:</p>
            <ul style="font-size: 0.85rem; margin: 0.5rem 0 0 1rem;">
                <li>Giảm độ lệch (skewness) của dữ liệu</li>
                <li>Làm cho phân phối gần chuẩn hơn</li>
                <li>Cải thiện hiệu suất mô hình</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col_transform1, col_transform2 = st.columns([1, 1])
        
        with col_transform1:
            st.markdown("##### ⚙️ Cấu Hình Biến Đổi")
            
            numeric_cols_for_transform = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols_for_transform:
                selected_transform_col = st.selectbox(
                    "Chọn cột cần biến đổi:",
                    numeric_cols_for_transform,
                    key="transform_col",
                    help="Chọn cột số để áp dụng biến đổi phân phối"
                )
                
                # Show distribution info
                col_data_transform = data[selected_transform_col].dropna()
                if len(col_data_transform) > 0:
                    skewness = col_data_transform.skew()
                    
                    stat_t1, stat_t2 = st.columns(2)
                    with stat_t1:
                        st.metric("Skewness", f"{skewness:.3f}")
                    with stat_t2:
                        if abs(skewness) < 0.5:
                            st.success("✅ Gần chuẩn")
                        elif abs(skewness) < 1.0:
                            st.warning("⚠️ Lệch vừa")
                        else:
                            st.error("❌ Lệch mạnh")
                    
                    st.markdown("---")
                    
                    # Transformation method selection
                    transform_method = st.selectbox(
                        "Phương pháp biến đổi:",
                        [
                            "Log (logarithm)",
                            "Log1p (log(1+x))",
                            "Sqrt (square root)",
                            "Cbrt (cube root)",
                            "Box-Cox",
                            "Yeo-Johnson",
                            "Reciprocal (1/x)",
                            "Square (x²)"
                        ],
                        key="transform_method",
                        help="Chọn phép biến đổi phù hợp với phân phối dữ liệu"
                    )
                    
                    # Show method description
                    method_desc = {
                        "Log (logarithm)": "Giảm skew dương, yêu cầu giá trị > 0",
                        "Log1p (log(1+x))": "Như Log nhưng xử lý được giá trị 0",
                        "Sqrt (square root)": "Giảm skew dương nhẹ hơn Log",
                        "Cbrt (cube root)": "Giảm skew dương, xử lý được giá trị âm",
                        "Box-Cox": "Tự động tìm λ tối ưu, yêu cầu giá trị > 0",
                        "Yeo-Johnson": "Như Box-Cox nhưng xử lý được giá trị âm",
                        "Reciprocal (1/x)": "Cho phân phối lệch phải mạnh",
                        "Square (x²)": "Tăng skew (ít dùng)"
                    }
                    
                    st.info(f"📝 {method_desc.get(transform_method, '')}")
                    
                    # Check if method is applicable
                    can_apply = True
                    warning_msg = ""
                    
                    if transform_method == "Log (logarithm)" and (col_data_transform <= 0).any():
                        can_apply = False
                        warning_msg = "⚠️ Log yêu cầu tất cả giá trị > 0"
                    elif transform_method == "Box-Cox" and (col_data_transform <= 0).any():
                        can_apply = False
                        warning_msg = "⚠️ Box-Cox yêu cầu tất cả giá trị > 0"
                    elif transform_method == "Reciprocal (1/x)" and (col_data_transform == 0).any():
                        can_apply = False
                        warning_msg = "⚠️ Reciprocal không xử lý được giá trị 0"
                    
                    if not can_apply:
                        st.warning(warning_msg)
                    
                    if st.button("✅ Áp Dụng Biến Đổi", key="apply_transform", use_container_width=True, type="primary", disabled=not can_apply):
                        with st.spinner(f"Đang biến đổi cột {selected_transform_col}..."):
                            # Backup
                            if 'transform_backup' not in st.session_state:
                                st.session_state.transform_backup = {}
                            st.session_state.transform_backup[selected_transform_col] = st.session_state.data[selected_transform_col].copy()
                            
                            # Apply transformation
                            if transform_method == "Log (logarithm)":
                                st.session_state.data[selected_transform_col] = np.log(st.session_state.data[selected_transform_col])
                            elif transform_method == "Log1p (log(1+x))":
                                st.session_state.data[selected_transform_col] = np.log1p(st.session_state.data[selected_transform_col])
                            elif transform_method == "Sqrt (square root)":
                                st.session_state.data[selected_transform_col] = np.sqrt(np.abs(st.session_state.data[selected_transform_col]))
                            elif transform_method == "Cbrt (cube root)":
                                st.session_state.data[selected_transform_col] = np.cbrt(st.session_state.data[selected_transform_col])
                            elif transform_method == "Box-Cox":
                                from scipy import stats
                                st.session_state.data[selected_transform_col], _ = stats.boxcox(st.session_state.data[selected_transform_col].dropna())
                            elif transform_method == "Yeo-Johnson":
                                from scipy import stats
                                st.session_state.data[selected_transform_col], _ = stats.yeojohnson(st.session_state.data[selected_transform_col].dropna())
                            elif transform_method == "Reciprocal (1/x)":
                                st.session_state.data[selected_transform_col] = 1 / st.session_state.data[selected_transform_col]
                            elif transform_method == "Square (x²)":
                                st.session_state.data[selected_transform_col] = np.square(st.session_state.data[selected_transform_col])
                            
                            st.success(f"✅ Đã áp dụng {transform_method} cho cột `{selected_transform_col}`!")
                            st.rerun()
                else:
                    st.warning("Cột này không có dữ liệu hợp lệ")
            else:
                st.info("Không có cột số nào để biến đổi")
        
        with col_transform2:
            st.markdown("##### 📊 Trực Quan Hóa Phân Phối")
            
            if numeric_cols_for_transform and 'selected_transform_col' in locals():
                # Show distribution plot
                col_data_viz = data[selected_transform_col].dropna()
                
                if len(col_data_viz) > 0:
                    fig = go.Figure()
                    
                    # Histogram
                    fig.add_trace(go.Histogram(
                        x=col_data_viz,
                        name='Distribution',
                        marker_color='#667eea',
                        opacity=0.7,
                        nbinsx=30
                    ))
                    
                    fig.update_layout(
                        title=f"Phân phối - {selected_transform_col}",
                        xaxis_title="Giá trị",
                        yaxis_title="Tần suất",
                        template="plotly_dark",
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.markdown("##### 📈 Thống Kê")
                    stats_df = pd.DataFrame({
                        'Thống kê': ['Mean', 'Median', 'Std', 'Min', 'Max', 'Skewness'],
                        'Giá trị': [
                            f"{col_data_viz.mean():.2f}",
                            f"{col_data_viz.median():.2f}",
                            f"{col_data_viz.std():.2f}",
                            f"{col_data_viz.min():.2f}",
                            f"{col_data_viz.max():.2f}",
                            f"{col_data_viz.skew():.3f}"
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Section 5: Mã Hóa Biến Phân Loại
        st.markdown("---")
        st.markdown("### 5️⃣ Mã Hóa Biến Phân Loại")
        
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
                
                # Show recommendation
                from backend.data_processing import recommend_encoding
                recommendation = recommend_encoding(data, selected_enc_col)
                
                st.markdown(f"""
                <div style="background-color: #1e3a5f; padding: 0.8rem; border-radius: 6px; border-left: 3px solid #3b82f6; margin: 0.5rem 0;">
                    <p style="margin: 0; font-size: 0.85rem;">
                        <strong>💡 Gợi ý:</strong> {recommendation['recommendation']}<br>
                        <span style="font-size: 0.8rem; opacity: 0.9;">{recommendation['reason']}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Encoding method selection
                encoding_method = st.selectbox(
                    "Phương pháp mã hóa:",
                    ["One-Hot Encoding", "Label Encoding", "Target Encoding", "Ordinal Encoding"],
                    key="encoding_method"
                )
                
                # Method-specific parameters
                encoding_params = {}
                
                if encoding_method == "One-Hot Encoding":
                    drop_first = st.checkbox(
                        "Drop first dummy (tránh multicollinearity)",
                        value=False,
                        key="onehot_drop_first",
                        help="Bỏ cột dummy đầu tiên để tránh hiện tượng đa cộng tuyến"
                    )
                    encoding_params['drop_first'] = drop_first
                
                elif encoding_method == "Target Encoding":
                    st.markdown("**Cấu hình Target Encoding:**")
                    
                    # Find target column
                    potential_targets = [col for col in data.columns 
                                       if 'target' in col.lower() or 'default' in col.lower() 
                                       or 'label' in col.lower() or 'churn' in col.lower()]
                    
                    numeric_cols_for_target = data.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if potential_targets:
                        default_target = potential_targets[0]
                    elif numeric_cols_for_target:
                        default_target = numeric_cols_for_target[-1]
                    else:
                        default_target = None
                    
                    if default_target and numeric_cols_for_target:
                        target_col = st.selectbox(
                            "Chọn cột target:",
                            numeric_cols_for_target,
                            index=numeric_cols_for_target.index(default_target) if default_target in numeric_cols_for_target else 0,
                            key="target_encoding_target",
                            help="Cột target để tính mean encoding"
                        )
                        
                        smoothing = st.slider(
                            "Smoothing (tránh overfitting):",
                            min_value=0.0,
                            max_value=10.0,
                            value=1.0,
                            step=0.5,
                            key="target_encoding_smoothing",
                            help="Giá trị cao hơn = ít overfitting hơn"
                        )
                        
                        encoding_params['target_column'] = target_col
                        encoding_params['smoothing'] = smoothing
                    else:
                        st.warning("⚠️ Không tìm thấy cột target. Vui lòng chọn phương pháp khác.")
                
                elif encoding_method == "Ordinal Encoding":
                    st.markdown("**Thứ tự các categories:**")
                    st.info("💡 Sắp xếp theo thứ tự có ý nghĩa (thấp → cao)")
                
                # Initialize encoding config
                if 'encoding_config' not in st.session_state:
                    st.session_state.encoding_config = {}
                
                # Add configuration button
                enc_btn_col1, enc_btn_col2 = st.columns(2)
                with enc_btn_col1:
                    if st.button("➕ Thêm Cấu Hình", key="add_enc_config", use_container_width=True):
                        st.session_state.encoding_config[selected_enc_col] = {
                            'method': encoding_method,
                            'unique_count': unique_count,
                            'params': encoding_params
                        }
                        st.success(f"✅ Đã thêm cấu hình cho `{selected_enc_col}`")
                        st.rerun()
                
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
                    with st.spinner("Đang mã hóa các biến phân loại..."):
                        try:
                            # Import backend encoder
                            from backend.data_processing import encode_categorical
                            
                            # Backup columns before encoding
                            if 'column_backups' not in st.session_state:
                                st.session_state.column_backups = {}
                            
                            for col in st.session_state.encoding_config.keys():
                                if col in st.session_state.data.columns:
                                    backup_key = f"encoding_{col}"
                                    st.session_state.column_backups[backup_key] = st.session_state.data[col].copy()
                            
                            encoded_data = st.session_state.data.copy()
                            all_encoding_info = {}
                            total_new_cols = 0
                            
                            # Apply each encoding configuration
                            for col, cfg in st.session_state.encoding_config.items():
                                method = cfg['method']
                                params = cfg.get('params', {})
                                
                                # Apply encoding for this column
                                encoded_data, encoding_info = encode_categorical(
                                    data=encoded_data,
                                    method=method,
                                    columns=[col],
                                    **params
                                )
                                
                                # Merge encoding info
                                all_encoding_info.update(encoding_info)
                                
                                # Count new columns (for One-Hot)
                                if 'new_columns' in encoding_info.get(col, {}):
                                    total_new_cols += encoding_info[col]['n_new_columns']
                            
                            # Save encoded data
                            st.session_state.data = encoded_data
                            
                            # Save encoding info to session
                            if 'encoding_applied_info' not in st.session_state:
                                st.session_state.encoding_applied_info = {}
                            st.session_state.encoding_applied_info.update(all_encoding_info)
                            
                            # Success message
                            st.success(f"✅ Đã mã hóa {len(st.session_state.encoding_config)} biến!")
                            
                            # Show summary
                            summary_items = []
                            for col, info in all_encoding_info.items():
                                if info['method'] == 'One-Hot Encoding':
                                    summary_items.append(f"- `{col}` → {info['n_new_columns']} cột mới")
                                else:
                                    summary_items.append(f"- `{col}` → {info['method']}")
                            
                            st.info("📊 **Kết quả mã hóa:**\n" + "\n".join(summary_items))
                            
                            # Mark configs as applied instead of clearing
                            for col in st.session_state.encoding_config:
                                st.session_state.encoding_config[col]['applied'] = True
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Lỗi khi mã hóa: {str(e)}")
                            import traceback
                            with st.expander("Chi tiết lỗi"):
                                st.code(traceback.format_exc())
                
                # Show applied encoding info if exists
                if st.session_state.get('encoding_applied_info'):
                    with st.expander("📋 Xem Chi Tiết Mã Hóa Đã Áp Dụng"):
                        for col, info in st.session_state.encoding_applied_info.items():
                            st.markdown(f"**{col}** - {info['method']}")
                            
                            if 'new_columns' in info:
                                st.write(f"Tạo {info['n_new_columns']} cột mới:", info['new_columns'][:10])
                            elif 'mapping' in info and len(str(info['mapping'])) < 500:
                                st.write("Mapping:", info['mapping'])
                            
                            st.markdown("---")
            
            else:
                st.info("💡 Chưa có cấu hình mã hóa nào. Hãy chọn cột và phương pháp ở trên.")
        
        else:
            st.success("✅ Không có biến phân loại cần mã hóa")
        
        # Section 6: Cân Bằng Dữ Liệu
        st.markdown("---")
        st.markdown("### 6️⃣ Cân Bằng Dữ Liệu")
        
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

