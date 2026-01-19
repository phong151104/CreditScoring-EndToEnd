"""
Login Page - Authentication UI
"""

import streamlit as st
from utils.session_state import init_session_state
from backend.auth import authenticate, ROLE_NAMES


def render():
    """Render login page"""
    init_session_state()
    
    # Custom CSS for login page
    st.markdown("""
    <style>
        .login-container {
            max-width: 450px;
            margin: 2rem auto;
            padding: 2.5rem;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .login-header h1 {
            color: #667eea;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .login-header p {
            color: #94a3b8;
            font-size: 0.95rem;
        }
        .role-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0.2rem;
        }
        .role-admin { background: #065f46; color: #10b981; }
        .role-validator { background: #1e3a5f; color: #3b82f6; }
        .role-scorer { background: #713f12; color: #f59e0b; }
        
        /* Custom Button Style for Login Form */
        div[data-testid="stForm"] .stButton button {
            background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stForm"] .stButton button:hover {
            background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
            transform: translateY(-1px) !important;
        }
        div[data-testid="stForm"] .stButton button:active {
            transform: translateY(0) !important;
        }
        
        /* Adjust password visibility toggle position */
        div[data-testid="stInputRightElement"] {
            right: 2px !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Centered login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-header">
            <h1>🏦 Credit Scoring System</h1>
            <p>Đăng nhập để tiếp tục</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "👤 Tên đăng nhập",
                placeholder="Nhập tên đăng nhập...",
                key="login_username"
            )
            
            password = st.text_input(
                "🔒 Mật khẩu",
                type="password",
                placeholder="Nhập mật khẩu...",
                key="login_password"
            )
            
            submit = st.form_submit_button("🔐 Đăng nhập", width='stretch')
        
        # Handle login
        if submit:
            if not username or not password:
                st.error("❌ Vui lòng nhập đầy đủ thông tin!")
            else:
                user = authenticate(username, password)
                if user:
                    # Set session state
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.user_role = user.role
                    st.session_state.login_time = st.session_state.get('_current_time', None)
                    
                    st.success(f"✅ Đăng nhập thành công! Xin chào {user.display_name}")
                    st.rerun()
                else:
                    st.error("❌ Sai tên đăng nhập hoặc mật khẩu!")
        

        
        # Role information
        st.markdown("---")
        
        with st.expander("ℹ️ Thông tin vai trò"):
            st.markdown("""
            **Hệ thống có 3 vai trò:**
            
            <span class="role-badge role-admin">👨‍💼 Admin</span>
            **Quản trị viên**
            - Full quyền truy cập mọi chức năng
            - Quản lý người dùng, cấu hình hệ thống
            
            <span class="role-badge role-admin" style="background:#8e44ad;">👷 Builder</span>
            **Xây dựng mô hình**
            - Upload data, EDA, Train model, Tuning
            - Không có quyền quản lý users
            
            <span class="role-badge role-validator">👨‍🔬 Validator</span>
            **Kiểm định & Đánh giá**
            - Xem kết quả training, EDA, SHAP
            - Phê duyệt hoặc từ chối model
            
            <span class="role-badge role-scorer">👨‍💻 Scorer</span>
            **Người dùng chấm điểm**
            - Chỉ sử dụng chức năng dự đoán
            - Xem giải thích kết quả
            """, unsafe_allow_html=True)
        
        # Demo accounts (for development)
        with st.expander("🔑 Tài khoản demo (Development)"):
            st.markdown("""
            | Username | Password | Role |
            |----------|----------|------|
            | `admin` | `admin123` | Admin |
            | `builder` | `builder123` | Model Builder |
            | `validator` | `validator123` | Validator |
            | `scorer` | `scorer123` | Scorer |
            """)


def logout():
    """Logout current user"""
    keys_to_clear = ['authenticated', 'user', 'user_role', 'login_time']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
