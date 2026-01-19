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

# Import utilities
from utils.ui_components import load_custom_css, render_header
from utils.session_state import init_session_state
from utils.permissions import (
    is_authenticated, 
    get_current_user, 
    get_current_role,
    get_allowed_pages,
    is_view_only,
    can_access_page
)

# Initialize session state
init_session_state()

# Add anchor at top of page for scroll reset
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

# CSS to help with scroll behavior
st.markdown("""
<style>
    /* Prevent scroll jump issues */
    .main .block-container {
        padding-top: 1rem;
    }
    /* Smooth scroll behavior */
    html {
        scroll-behavior: smooth;
    }
    /* Auto scroll to top on page load */
    .stApp {
        scroll-margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Load CSS tùy chỉnh
try:
    load_custom_css()
except Exception as e:
    print(f"[X] CSS error: {e}", file=sys.stderr)

# ============================================
# AUTHENTICATION CHECK
# ============================================
if not is_authenticated():
    # Show login page
    from views import login
    login.render()
    st.stop()

# ============================================
# AUTHENTICATED USER - Show main app
# ============================================

# Get current user info
current_user = get_current_user()
current_role = get_current_role()

# Render header
try:
    render_header()
except Exception as e:
    print(f"[X] Header error: {e}", file=sys.stderr)
    st.markdown("# CREDIT SCORING SYSTEM")
    st.markdown("### Advanced Risk Assessment & Prediction Platform")
    st.markdown("---")

# Sidebar navigation
with st.sidebar:
    # User info section
    role_colors = {
        'admin': '#10b981',
        'validator': '#3b82f6', 
        'scorer': '#f59e0b'
    }
    role_icons = {
        'admin': '👨‍💼',
        'validator': '👨‍🔬',
        'scorer': '👨‍💻'
    }
    role_color = role_colors.get(current_role, '#667eea')
    role_icon = role_icons.get(current_role, '👤')
    
    st.markdown(f"""
    <div style='text-align: left; padding: 1rem 0;'>
        <h2 style='margin: 0; color: #667eea; font-weight: 600;'>
            <span style='font-size: 1.8rem;'>🏦</span> Credit Scoring
        </h2>
        <p style='margin: 0.3rem 0 0 0; color: #aaa; font-size: 0.85rem;'>Risk Assessment Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User info box
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0; 
                border-left: 3px solid {role_color};'>
        <div style='display: flex; align-items: center; gap: 0.5rem;'>
            <span style='font-size: 1.5rem;'>{role_icon}</span>
            <div>
                <div style='color: white; font-weight: 600;'>{current_user.display_name}</div>
                <div style='color: {role_color}; font-size: 0.8rem;'>{current_role.upper()}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Logout button
    if st.button("🚪 Đăng xuất", key="logout_btn", width='stretch'):
        from views.login import logout
        logout()
        st.rerun()
    
    st.markdown("---")
    
    # Menu điều hướng
    st.markdown("### NAVIGATION")
    
    # Get navigation options based on role
    nav_options = get_allowed_pages()
    
    # Get default index from session state if set
    default_index = 0
    if 'nav_page' in st.session_state and st.session_state.nav_page in nav_options:
        default_index = nav_options.index(st.session_state.nav_page)
        # Clear nav_page after using it
        del st.session_state.nav_page
    
    # Navigation with radio buttons
    page = st.radio(
        "Chọn chức năng:",
        nav_options,
        index=default_index,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Thông tin phiên làm việc
    with st.expander("▼ Session Status"):
        if 'data' in st.session_state and st.session_state.data is not None:
            st.success(f"● Đã tải dữ liệu: {len(st.session_state.data)} dòng")
        else:
            st.info("○ Chưa tải dữ liệu")
        
        if 'model' in st.session_state and st.session_state.model is not None:
            st.success("● Đã huấn luyện mô hình")
        else:
            st.info("○ Chưa huấn luyện mô hình")
        
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
            st.success(f"● Đã xử lý: {len(st.session_state.processed_data)} dòng")
    
    st.markdown("---")
    st.caption("© 2025 Credit Scoring System v2.0")

# Định tuyến trang với logging
# Track page change to handle scroll
if 'current_page' not in st.session_state:
    st.session_state.current_page = page
    
page_changed = st.session_state.current_page != page
if page_changed:
    st.session_state.current_page = page
    # Add JavaScript to scroll to top when page changes
    st.markdown("""
    <script>
        window.parent.document.querySelector('section.main').scrollTo(0, 0);
    </script>
    """, unsafe_allow_html=True)

# Store view_only status for current page
st.session_state.view_only_mode = is_view_only(page)

try:
    if page == "🏠 Dashboard":
        from views import home
        home.render()
    elif page == "📊 Data Upload & Analysis":
        from views import upload_eda
        upload_eda.render()
    elif page == "⚙️ Feature Engineering":
        from views import feature_engineering
        feature_engineering.render()
    elif page == "🧠 Model Training":
        from views import model_training
        model_training.render()
    elif page == "💡 Model Explanation":
        from views import shap_explanation
        shap_explanation.render()
    elif page == "🎯 Prediction & Advisory":
        from views import prediction
        prediction.render()
    elif page == "⚡ Admin Settings":
        from views import admin_settings
        admin_settings.render()
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    st.error(f"Error loading page: {e}")
    st.error("Check terminal for details")
