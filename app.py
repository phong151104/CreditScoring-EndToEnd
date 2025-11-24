"""
Credit Scoring System - Main Application
Advanced Risk Assessment & Prediction Platform
"""

import streamlit as st
from pathlib import Path
import sys

# Enable debug logging
sys.stdout.flush()

# Cấu hình trang
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom CSS
from utils.ui_components import load_custom_css, render_header
import sys

# Enable logging

# Load CSS tùy chỉnh
try:
    load_custom_css()
except Exception as e:
    print(f"✗ CSS error: {e}", file=sys.stderr)

# Render header
try:
    render_header()
except Exception as e:
    print(f"✗ Header error: {e}", file=sys.stderr)
    st.markdown("# CREDIT SCORING SYSTEM")
    st.markdown("### Advanced Risk Assessment & Prediction Platform")
    st.markdown("---")

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align: left; padding: 1rem 0;'>
        <h2 style='margin: 0; color: #667eea; font-weight: 600;'>
            <span style='font-size: 1.8rem;'>▣</span> Credit Scoring
        </h2>
        <p style='margin: 0.3rem 0 0 0; color: #aaa; font-size: 0.85rem;'>Risk Assessment Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Menu điều hướng
    st.markdown("### NAVIGATION")
    
    # Navigation with radio buttons
    page = st.radio(
        "Select function:",
        ["◉ Dashboard", "↑ Data Upload & Analysis", "⚡ Feature Engineering", 
         "◈ Model Training", "◐ Model Explanation", "◎ Prediction & Advisory"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Thông tin phiên làm việc
    with st.expander("▼ Session Status"):
        if 'data' in st.session_state and st.session_state.data is not None:
            st.success(f"● Data loaded: {len(st.session_state.data)} rows")
        else:
            st.info("○ No data uploaded")
        
        if 'model' in st.session_state and st.session_state.model is not None:
            st.success("● Model trained")
        else:
            st.info("○ No model trained")
        
        # Show configurations count
        total_configs = (
            len(st.session_state.get('missing_config', {})) +
            len(st.session_state.get('encoding_config', {})) +
            len(st.session_state.get('binning_config', {}))
        )
        if total_configs > 0:
            st.success(f"● {total_configs} cấu hình đã lưu")
            
            # Show breakdown
            if len(st.session_state.get('missing_config', {})) > 0:
                st.caption(f"  - Missing: {len(st.session_state.missing_config)}")
            if len(st.session_state.get('encoding_config', {})) > 0:
                st.caption(f"  - Encoding: {len(st.session_state.encoding_config)}")
            if len(st.session_state.get('binning_config', {})) > 0:
                st.caption(f"  - Binning: {len(st.session_state.binning_config)}")
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            st.success(f"● Processed: {len(st.session_state.processed_data)} rows")
    
    st.markdown("---")
    st.caption("© 2025 Credit Scoring System v1.0")

# Định tuyến trang với logging

try:
    if page == "◉ Dashboard":
        from views import home
        home.render()
    elif page == "↑ Data Upload & Analysis":
        from views import upload_eda
        upload_eda.render()
    elif page == "⚡ Feature Engineering":
        from views import feature_engineering
        feature_engineering.render()
    elif page == "◈ Model Training":
        from views import model_training
        model_training.render()
    elif page == "◐ Model Explanation":
        from views import shap_explanation
        shap_explanation.render()
    elif page == "◎ Prediction & Advisory":
        from views import prediction
        prediction.render()
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    st.error(f"Error loading page: {e}")
    st.error("Check terminal for details")

