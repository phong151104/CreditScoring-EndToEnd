"""
Permissions Module - Role-Based Access Control (RBAC)
"""

import streamlit as st
from functools import wraps
from typing import List, Callable

# Role permission mappings
ROLE_PERMISSIONS = {
    'admin': [
        'upload_data',
        'view_eda',
        'analyze_ai',
        'feature_engineering',
        'model_training',
        'model_tuning',
        'view_shap_global',
        'view_shap_local',
        'init_shap',
        'prediction',
        'admin_settings',
        'user_management',
        'export_data',
        'configure_thresholds'
    ],
    'model_builder': [
        'upload_data',
        'view_eda',
        'analyze_ai',
        'feature_engineering',
        'model_training',
        'model_tuning',
        'view_shap_global',
        'view_shap_local',
        'init_shap',
        'prediction',
        'export_data',
        'configure_thresholds'
        # Note: NO 'admin_settings' and 'user_management'
    ],
    'validator': [
        'view_eda',
        'view_feature_config',
        'view_training_results',
        'view_shap_global',
        'view_shap_local',
        'add_comments',
        'export_reports'
    ],
    'scorer': [
        'prediction',
        'export_prediction_report'
    ]
}

# Page access by role
PAGE_ACCESS = {
    'admin': [
        '🏠 Dashboard',
        '📊 Data Upload & Analysis',
        '⚙️ Feature Engineering',
        '🧠 Model Training',
        '💡 Model Explanation',
        '🎯 Prediction & Advisory',
        '⚡ Admin Settings'
    ],
    'model_builder': [
        '🏠 Dashboard',
        '📊 Data Upload & Analysis',
        '⚙️ Feature Engineering',
        '🧠 Model Training',
        '💡 Model Explanation',
        '🎯 Prediction & Advisory'
        # Note: NO '⚡ Admin Settings'
    ],
    'validator': [
        '🏠 Dashboard',
        '📊 Data Upload & Analysis',
        '⚙️ Feature Engineering',
        '🧠 Model Training',
        '💡 Model Explanation',
        '🎯 Prediction & Advisory'
    ],
    'scorer': [
        '🎯 Prediction & Advisory'
    ]
}

# View mode for pages (for validator/scorer)
VIEW_ONLY_PAGES = {
    'validator': [
        '📊 Data Upload & Analysis',
        '⚙️ Feature Engineering',
        '🧠 Model Training',
        '💡 Model Explanation',
        '🎯 Prediction & Advisory'
    ],
    'scorer': [
        '💡 Model Explanation'
    ]
    # Note: model_builder has FULL access, not view-only
}


def has_permission(permission: str) -> bool:
    """Check if current user has a specific permission"""
    if not is_authenticated():
        return False
    
    role = get_current_role()
    return permission in ROLE_PERMISSIONS.get(role, [])


def has_any_permission(permissions: List[str]) -> bool:
    """Check if current user has any of the specified permissions"""
    return any(has_permission(p) for p in permissions)


def has_all_permissions(permissions: List[str]) -> bool:
    """Check if current user has all of the specified permissions"""
    return all(has_permission(p) for p in permissions)


def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)


def get_current_user():
    """Get current authenticated user"""
    return st.session_state.get('user', None)


def get_current_role() -> str:
    """Get current user's role"""
    return st.session_state.get('user_role', None)


def get_allowed_pages() -> List[str]:
    """Get list of pages current user can access"""
    role = get_current_role()
    if role is None:
        return []
    return PAGE_ACCESS.get(role, [])


def is_view_only(page: str) -> bool:
    """Check if current user has view-only access to a page"""
    role = get_current_role()
    if role == 'admin':
        return False
    return page in VIEW_ONLY_PAGES.get(role, [])


def can_access_page(page: str) -> bool:
    """Check if current user can access a page"""
    role = get_current_role()
    if role is None:
        return False
    return page in PAGE_ACCESS.get(role, [])


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            st.warning("⚠️ Vui lòng đăng nhập để tiếp tục.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper


def require_permission(permission: str) -> Callable:
    """Decorator factory to require a specific permission"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_authenticated():
                st.warning("⚠️ Vui lòng đăng nhập để tiếp tục.")
                st.stop()
            if not has_permission(permission):
                st.error(f"❌ Bạn không có quyền thực hiện chức năng này.")
                st.stop()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(roles: List[str]) -> Callable:
    """Decorator factory to require one of specified roles"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_authenticated():
                st.warning("⚠️ Vui lòng đăng nhập để tiếp tục.")
                st.stop()
            if get_current_role() not in roles:
                st.error(f"❌ Chức năng này chỉ dành cho: {', '.join(roles)}")
                st.stop()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def show_view_only_banner():
    """Display a banner indicating view-only mode"""
    st.info("👁️ **Chế độ xem** - Bạn chỉ có quyền xem nội dung này, không thể chỉnh sửa.")


def show_no_permission_message(action: str = "thực hiện chức năng này"):
    """Display a message when user doesn't have permission"""
    st.warning(f"⚠️ Bạn không có quyền {action}.")


def check_and_show_view_only(page: str) -> bool:
    """Check if view-only and show banner if so. Returns True if view-only."""
    if is_view_only(page):
        show_view_only_banner()
        return True
    return False
