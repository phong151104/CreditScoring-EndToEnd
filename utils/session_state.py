"""
Session State Management - Quản lý trạng thái phiên làm việc
"""

import streamlit as st

def init_session_state():
    """Khởi tạo session state"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    
    if 'explainer' not in st.session_state:
        st.session_state.explainer = None
    
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = None
    
    # Feature Engineering configurations - persist across page changes
    if 'missing_config' not in st.session_state:
        st.session_state.missing_config = {}
    
    if 'encoding_config' not in st.session_state:
        st.session_state.encoding_config = {}
    
    if 'scaling_config' not in st.session_state:
        st.session_state.scaling_config = {}
    
    if 'outlier_config' not in st.session_state:
        st.session_state.outlier_config = {}
    
    if 'binning_config' not in st.session_state:
        st.session_state.binning_config = {}
    
    # AI Analysis cache
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = None
    
    if 'eda_summary' not in st.session_state:
        st.session_state.eda_summary = None

def clear_session_state():
    """Xóa toàn bộ session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def clear_data_related_state():
    """Xóa các state liên quan đến data khi upload file mới"""
    keys_to_clear = [
        'data', 
        'processed_data', 
        'selected_features', 
        'model', 
        'model_type', 
        'model_metrics', 
        'explainer', 
        'shap_values',
        'prediction_input',
        'prediction_result',
        # Clear all configurations
        'missing_config',
        'encoding_config',
        'scaling_config',
        'outlier_config',
        'binning_config',
        'ai_analysis',
        'eda_summary'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            if key == 'selected_features':
                st.session_state[key] = []
            elif key in ['model_metrics', 'missing_config', 'encoding_config', 
                        'scaling_config', 'outlier_config', 'binning_config']:
                st.session_state[key] = {}
            else:
                st.session_state[key] = None

def get_session_info():
    """Lấy thông tin về session hiện tại"""
    info = {
        "has_data": st.session_state.data is not None,
        "has_processed_data": st.session_state.processed_data is not None,
        "has_model": st.session_state.model is not None,
        "num_features": len(st.session_state.selected_features) if st.session_state.selected_features else 0,
        "num_missing_configs": len(st.session_state.get('missing_config', {})),
        "num_encoding_configs": len(st.session_state.get('encoding_config', {})),
        "num_binning_configs": len(st.session_state.get('binning_config', {})),
        "has_ai_analysis": st.session_state.get('ai_analysis') is not None,
    }
    return info


def print_session_debug():
    """In thông tin debug session state ra console"""
    info = get_session_info()
    print("=" * 50)
    print("SESSION STATE DEBUG INFO")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key:.<30} {value}")
    print("=" * 50)

