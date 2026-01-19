"""
Admin Settings Page - System configuration for Admins
"""

import streamlit as st
import pandas as pd
from utils.session_state import init_session_state
from utils.permissions import require_role, has_permission
from backend.auth import get_all_users, create_user, update_user, delete_user, ROLES, ROLE_NAMES


def render():
    """Render admin settings page"""
    init_session_state()
    
    st.markdown("## ⚡ Cài đặt Hệ thống")
    st.markdown("Quản lý người dùng và phân quyền hệ thống.")
    
    # Render User Management directly
    _render_user_management()


def _render_user_management():
    """Render user management section"""
    st.markdown("### 👥 Quản lý Người dùng")
    
    # Get all users
    users = get_all_users()
    
    # Display users table
    if users:
        users_data = []
        for user in users:
            users_data.append({
                'Username': user.username,
                'Display Name': user.display_name,
                'Role': ROLE_NAMES.get(user.role, user.role),
                'Role Code': user.role
            })
        
        df = pd.DataFrame(users_data)
        st.dataframe(df[['Username', 'Display Name', 'Role']], width='stretch', hide_index=True)
    else:
        st.info("Không có users trong hệ thống.")
    
    st.markdown("---")
    
    # Add new user form
    with st.expander("➕ Thêm User mới"):
        with st.form("add_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username", placeholder="username")
                new_display_name = st.text_input("Display Name", placeholder="Tên hiển thị")
            
            with col2:
                new_password = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", options=ROLES, format_func=lambda x: ROLE_NAMES.get(x, x))
            
            submitted = st.form_submit_button("➕ Thêm User", type="primary")
            
            if submitted:
                if not new_username or not new_password:
                    st.error("❌ Username và Password là bắt buộc!")
                else:
                    user = create_user(new_username, new_password, new_role, new_display_name)
                    if user:
                        st.success(f"✅ Đã thêm user: {new_username}")
                        st.rerun()
                    else:
                        st.error("❌ Không thể tạo user. Username có thể đã tồn tại.")
    
    # Edit/Delete user
    with st.expander("✏️ Sửa/Xóa User"):
        if users:
            selected_username = st.selectbox(
                "Chọn user",
                options=[u.username for u in users],
                key="edit_user_select"
            )
            
            selected_user = next((u for u in users if u.username == selected_username), None)
            
            if selected_user:
                with st.form("edit_user_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        edit_display_name = st.text_input("Display Name", value=selected_user.display_name)
                        edit_password = st.text_input("New Password (leave empty to keep)", type="password")
                    
                    with col2:
                        current_role_idx = ROLES.index(selected_user.role) if selected_user.role in ROLES else 0
                        edit_role = st.selectbox(
                            "Role", 
                            options=ROLES, 
                            index=current_role_idx,
                            format_func=lambda x: ROLE_NAMES.get(x, x)
                        )
                    
                    col_save, col_delete = st.columns(2)
                    
                    with col_save:
                        save_btn = st.form_submit_button("💾 Lưu thay đổi", type="primary")
                    
                    with col_delete:
                        delete_btn = st.form_submit_button("🗑️ Xóa User")
                    
                    if save_btn:
                        user = update_user(
                            selected_username,
                            password=edit_password if edit_password else None,
                            role=edit_role,
                            display_name=edit_display_name
                        )
                        if user:
                            st.success(f"✅ Đã cập nhật user: {selected_username}")
                            st.rerun()
                        else:
                            st.error("❌ Không thể cập nhật user.")
                    
                    if delete_btn:
                        if selected_username == 'admin':
                            st.error("❌ Không thể xóa tài khoản admin!")
                        else:
                            if delete_user(selected_username):
                                st.success(f"✅ Đã xóa user: {selected_username}")
                                st.rerun()
                            else:
                                st.error("❌ Không thể xóa user.")


def _render_threshold_settings():
    """Render decision threshold settings"""
    st.markdown("### 📊 Cấu hình Ngưỡng Quyết định")
    
    # Initialize threshold settings
    if 'threshold_settings' not in st.session_state:
        st.session_state.threshold_settings = {
            'approve_pd_max': 0.15,
            'consider_pd_max': 0.35,
            'approve_score_min': 650,
            'consider_score_min': 500
        }
    
    settings = st.session_state.threshold_settings
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <p style='margin: 0; color: #94a3b8;'>
            Cấu hình ngưỡng để phân loại quyết định: <span style='color: #10b981;'>Phê duyệt</span> / 
            <span style='color: #f59e0b;'>Xem xét</span> / <span style='color: #ef4444;'>Từ chối</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📉 Ngưỡng PD (Probability of Default)")
        
        approve_pd = st.slider(
            "PD tối đa để PHÊ DUYỆT",
            min_value=0.0,
            max_value=0.5,
            value=settings['approve_pd_max'],
            step=0.01,
            format="%.2f",
            help="Nếu PD ≤ giá trị này → Phê duyệt"
        )
        
        consider_pd = st.slider(
            "PD tối đa để XEM XÉT",
            min_value=0.0,
            max_value=0.7,
            value=settings['consider_pd_max'],
            step=0.01,
            format="%.2f",
            help="Nếu PD ≤ giá trị này → Xem xét (nếu > approve_pd)"
        )
    
    with col2:
        st.markdown("#### 📈 Ngưỡng Credit Score")
        
        approve_score = st.slider(
            "Score tối thiểu để PHÊ DUYỆT",
            min_value=300,
            max_value=850,
            value=settings['approve_score_min'],
            step=10,
            help="Nếu Score ≥ giá trị này → Phê duyệt"
        )
        
        consider_score = st.slider(
            "Score tối thiểu để XEM XÉT",
            min_value=300,
            max_value=850,
            value=settings['consider_score_min'],
            step=10,
            help="Nếu Score ≥ giá trị này → Xem xét (nếu < approve_score)"
        )
    
    # Preview logic
    st.markdown("---")
    st.markdown("#### 🔍 Preview Logic")
    
    st.markdown(f"""
    | Quyết định | Điều kiện PD | Điều kiện Score |
    |------------|--------------|-----------------|
    | ✅ **Phê duyệt** | PD ≤ {approve_pd:.2f} | Score ≥ {approve_score} |
    | ⚠️ **Xem xét** | {approve_pd:.2f} < PD ≤ {consider_pd:.2f} | {consider_score} ≤ Score < {approve_score} |
    | ❌ **Từ chối** | PD > {consider_pd:.2f} | Score < {consider_score} |
    """)
    
    # Save button
    if st.button("💾 Lưu Cấu hình", type="primary"):
        st.session_state.threshold_settings = {
            'approve_pd_max': approve_pd,
            'consider_pd_max': consider_pd,
            'approve_score_min': approve_score,
            'consider_score_min': consider_score
        }
        st.success("✅ Đã lưu cấu hình ngưỡng quyết định!")


def _render_score_formula_settings():
    """Render credit score formula settings"""
    st.markdown("### 🔢 Công thức Credit Score")
    
    # Initialize formula settings
    if 'score_formula' not in st.session_state:
        st.session_state.score_formula = {
            'base_score': 600,
            'pdo': 20,
            'base_odds': 50,
            'min_score': 300,
            'max_score': 850
        }
    
    formula = st.session_state.score_formula
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <p style='margin: 0; color: #94a3b8;'>
            Công thức tính Credit Score theo chuẩn Basel II/III:<br>
            <code style='color: #667eea;'>Score = Base Score - PDO × log(Odds) / log(2)</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_score = st.number_input(
            "Base Score",
            min_value=400,
            max_value=800,
            value=formula['base_score'],
            step=10,
            help="Điểm cơ sở tại base odds"
        )
        
        pdo = st.number_input(
            "PDO (Points to Double Odds)",
            min_value=10,
            max_value=50,
            value=formula['pdo'],
            step=5,
            help="Số điểm giảm khi odds tăng gấp đôi"
        )
        
        base_odds = st.number_input(
            "Base Odds",
            min_value=10,
            max_value=100,
            value=formula['base_odds'],
            step=5,
            help="Tỷ lệ odds cơ sở (Good:Bad)"
        )
    
    with col2:
        min_score = st.number_input(
            "Min Score",
            min_value=0,
            max_value=500,
            value=formula['min_score'],
            step=10,
            help="Điểm tối thiểu có thể đạt được"
        )
        
        max_score = st.number_input(
            "Max Score",
            min_value=500,
            max_value=1000,
            value=formula['max_score'],
            step=10,
            help="Điểm tối đa có thể đạt được"
        )
    
    # Save button
    if st.button("💾 Lưu Công thức", type="primary", key="save_formula"):
        st.session_state.score_formula = {
            'base_score': base_score,
            'pdo': pdo,
            'base_odds': base_odds,
            'min_score': min_score,
            'max_score': max_score
        }
        st.success("✅ Đã lưu công thức Credit Score!")


def _render_export_import():
    """Render export/import settings"""
    st.markdown("### 📥 Export/Import Cấu hình")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Export")
        
        if st.button("📤 Export Settings", width='stretch'):
            import json
            
            export_data = {
                'threshold_settings': st.session_state.get('threshold_settings', {}),
                'score_formula': st.session_state.get('score_formula', {}),
                'export_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            st.download_button(
                "📥 Download JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name="credit_scoring_settings.json",
                mime="application/json",
                width='stretch'
            )
    
    with col2:
        st.markdown("#### 📥 Import")
        
        uploaded_file = st.file_uploader("Upload Settings JSON", type=['json'])
        
        if uploaded_file is not None:
            try:
                import json
                import_data = json.load(uploaded_file)
                
                st.json(import_data)
                
                if st.button("📥 Apply Settings", type="primary", width='stretch'):
                    if 'threshold_settings' in import_data:
                        st.session_state.threshold_settings = import_data['threshold_settings']
                    if 'score_formula' in import_data:
                        st.session_state.score_formula = import_data['score_formula']
                    
                    st.success("✅ Đã import settings thành công!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Lỗi khi đọc file: {str(e)}")
