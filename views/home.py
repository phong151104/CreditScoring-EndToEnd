"""
Dashboard - Home Page
"""

import streamlit as st
from utils.ui_components import render_info_card
from utils.session_state import init_session_state, get_session_info

def render():
    """Render dashboard page"""
    init_session_state()
    
    session_info = get_session_info()
    
    # Check if any preprocessing has been applied (any modification to data)
    # This includes any applied configs or transformations
    has_preprocessing = (
        # Config-based checks
        st.session_state.get('missing_config') or 
        st.session_state.get('encoding_config') or 
        st.session_state.get('scaling_config') or
        st.session_state.get('outlier_config') or
        st.session_state.get('validation_config') or
        st.session_state.get('binning_config') or
        st.session_state.get('balancing_config') or
        # Applied state checks
        st.session_state.get('balance_info') or  # Balancing đã áp dụng
        st.session_state.get('applied_missing') or
        st.session_state.get('applied_encoding') or
        st.session_state.get('applied_scaling') or
        st.session_state.get('applied_outlier') or
        st.session_state.get('applied_binning') or
        st.session_state.get('applied_validation') or
        # General processed data check
        session_info.get('has_processed_data', False)
    )
    
    # Define workflow steps
    workflow_steps = [
        {
            "name": "Tải dữ liệu",
            "icon": "📤",
            "done": session_info['has_data'],
            "detail": f"{st.session_state.data.shape[0]:,} dòng" if session_info['has_data'] else "Chưa tải"
        },
        {
            "name": "Chia tập dữ liệu",
            "icon": "✂️",
            "done": st.session_state.get('split_config') is not None,
            "detail": f"{st.session_state.split_config.get('train_ratio', 0)}%/{st.session_state.split_config.get('valid_ratio', 0)}%/{st.session_state.split_config.get('test_ratio', 0)}%" if st.session_state.get('split_config') else "Chưa chia"
        },
        {
            "name": "Tiền xử lý",
            "icon": "🔧",
            "done": has_preprocessing,
            "detail": "Đã xử lý" if has_preprocessing else "Chưa xử lý"
        },
        {
            "name": "Chọn đặc trưng",
            "icon": "🎯",
            "done": session_info['num_features'] > 0,
            "detail": f"{session_info['num_features']} features" if session_info['num_features'] > 0 else "Chưa chọn"
        },
        {
            "name": "Huấn luyện",
            "icon": "🧠",
            "done": session_info['has_model'],
            "detail": (st.session_state.get('selected_model_name') or st.session_state.get('model_type') or 'Đã train')[:12] if session_info['has_model'] else "Chưa train"
        },
        {
            "name": "Chọn model",
            "icon": "🏆",
            "done": st.session_state.get('selected_model_idx') is not None,
            "detail": f"AUC: {st.session_state.model_metrics.get('auc', 0):.2f}" if st.session_state.get('model_metrics') else "Chưa chọn"
        },
        {
            "name": "SHAP",
            "icon": "💡",
            "done": st.session_state.get('shap_explainer_obj') is not None or st.session_state.get('shap_values_computed') is not None,
            "detail": "Đã phân tích" if (st.session_state.get('shap_explainer_obj') or st.session_state.get('shap_values_computed')) else "Chưa phân tích"
        },
        {
            "name": "Dự đoán",
            "icon": "🎯",
            "done": st.session_state.get('prediction_result') is not None,
            "detail": f"Score: {st.session_state.prediction_result.get('credit_score', 'N/A')}" if st.session_state.get('prediction_result') else "Chưa dự đoán"
        },
    ]
    
    # Calculate progress
    completed_count = sum(1 for step in workflow_steps if step['done'])
    total_count = len(workflow_steps)
    progress_pct = int(completed_count / total_count * 100)
    
    # Find current step
    current_step_idx = -1
    for i, step in enumerate(workflow_steps):
        if not step['done']:
            current_step_idx = i
            break
    
    # Build all steps HTML first - single line to avoid rendering issues
    steps_html = ""
    for i, step in enumerate(workflow_steps):
        is_done = step['done']
        is_current = (i == current_step_idx)
        status_class = "done" if is_done else ("current" if is_current else "pending")
        check_icon = "✅" if is_done else ("🔄" if is_current else "○")
        
        steps_html += f'<div class="step-item {status_class}"><span class="step-icon">{step["icon"]}</span><span class="step-name">{step["name"]}</span><span class="step-detail">{step["detail"]}</span><span class="step-check">{check_icon}</span></div>'
    
    # Render everything in ONE markdown call
    st.markdown(f"""
    <style>
    .progress-container {{
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
    }}
    .progress-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }}
    .progress-title {{
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
    }}
    .progress-count {{
        font-size: 0.9rem;
        color: #667eea;
        font-weight: 600;
    }}
    .progress-bar-bg {{
        height: 10px;
        background: #374151;
        border-radius: 5px;
        overflow: hidden;
        margin-bottom: 1rem;
    }}
    .progress-bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 5px;
        transition: width 0.5s ease;
    }}
    .workflow-steps {{
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-between;
        gap: 0.5rem;
        flex-wrap: nowrap;
    }}
    .step-item {{
        flex: 1;
        text-align: center;
        padding: 0.6rem 0.3rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        min-width: 0;
    }}
    .step-item.done {{
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .step-item.pending {{
        background: rgba(71, 85, 105, 0.2);
        border: 1px solid rgba(71, 85, 105, 0.3);
    }}
    .step-item.current {{
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        animation: glow 2s infinite;
    }}
    @keyframes glow {{
        0%, 100% {{ box-shadow: 0 0 5px rgba(59, 130, 246, 0.3); }}
        50% {{ box-shadow: 0 0 15px rgba(59, 130, 246, 0.5); }}
    }}
    .step-icon {{
        font-size: 1.7rem;
        display: block;
        margin-bottom: 0.3rem;
    }}
    .step-name {{
        font-size: 0.85rem;
        color: #e2e8f0;
        font-weight: 600;
        display: block;
    }}
    .step-detail {{
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.2rem;
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .step-check {{
        font-size: 0.9rem;
        margin-top: 0.2rem;
        display: block;
    }}
    </style>
    
    <div class="progress-container">
        <div class="progress-header">
            <span class="progress-title">📋 Tiến độ Workflow</span>
            <span class="progress-count">{completed_count}/{total_count} bước ({progress_pct}%)</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {progress_pct}%;"></div>
        </div>
        <div class="workflow-steps">
            {steps_html}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Grid - 2 columns, 3 rows for 5 features + 1 empty
    st.markdown("### 🚀 Bắt đầu sử dụng")
    
    features = [
        {
            "icon": "📤",
            "title": "Tải Dữ Liệu & Phân Tích",
            "desc": "Nhập CSV, khám phá thống kê mô tả, phân phối và tương quan",
            "nav_key": "📊 Data Upload & Analysis",
            "color": "#3b82f6"
        },
        {
            "icon": "⚙️",
            "title": "Xử Lý Đặc Trưng", 
            "desc": "Xử lý missing, outliers, encoding, scaling và chia tập",
            "nav_key": "⚙️ Feature Engineering",
            "color": "#8b5cf6"
        },
        {
            "icon": "🧠",
            "title": "Huấn Luyện Mô Hình",
            "desc": "Train với Logistic, XGBoost, LightGBM, CatBoost",
            "nav_key": "🧠 Model Training",
            "color": "#10b981"
        },
        {
            "icon": "💡",
            "title": "Giải Thích Mô Hình",
            "desc": "Phân tích SHAP, feature importance và force plot",
            "nav_key": "💡 Model Explanation",
            "color": "#f59e0b"
        },
        {
            "icon": "🎯",
            "title": "Dự Đoán & Tư Vấn",
            "desc": "Dự đoán điểm tín dụng và nhận tư vấn AI",
            "nav_key": "🎯 Prediction & Advisory",
            "color": "#ef4444"
        },
    ]
    
    # Create 2 columns for features
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(features):
        target_col = col1 if i % 2 == 0 else col2
        
        with target_col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                        padding: 1.2rem; border-radius: 12px; margin-bottom: 0.8rem; 
                        border-left: 4px solid {feature['color']}; 
                        transition: all 0.3s ease;">
                <h4 style="margin: 0; color: {feature['color']}; font-size: 1.05rem;">{feature['icon']} {feature['title']}</h4>
                <p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 0.95rem;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Đi đến {feature['title']}", key=f"nav_{feature['title']}", width='stretch'):
                st.session_state.nav_page = feature['nav_key']
                st.rerun()

