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
    
    if 'selected_model_name' not in st.session_state:
        st.session_state.selected_model_name = None
    
    if 'selected_model_timestamp' not in st.session_state:
        st.session_state.selected_model_timestamp = None
    
    if 'selected_model_idx' not in st.session_state:
        st.session_state.selected_model_idx = None
    
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    
    if 'explainer' not in st.session_state:
        st.session_state.explainer = None
    
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = None
    
    # SHAP computed data
    if 'shap_explainer_obj' not in st.session_state:
        st.session_state.shap_explainer_obj = None
    
    if 'shap_values_computed' not in st.session_state:
        st.session_state.shap_values_computed = None
    
    if 'shap_X_explained' not in st.session_state:
        st.session_state.shap_X_explained = None
    
    if 'shap_feature_importance' not in st.session_state:
        st.session_state.shap_feature_importance = None
    
    if 'shap_expected_value' not in st.session_state:
        st.session_state.shap_expected_value = None
    
    # SHAP AI Chat
    if 'shap_chat_history' not in st.session_state:
        st.session_state.shap_chat_history = []
    
    if 'last_ai_analysis' not in st.session_state:
        st.session_state.last_ai_analysis = None
    
    if 'sample_question_selected' not in st.session_state:
        st.session_state.sample_question_selected = None
    
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
        # Data
        'data', 
        'processed_data', 
        'selected_features',
        'target_column',
        
        # Train/Valid/Test split
        'train_data',
        'valid_data', 
        'test_data',
        'X_train', 'X_valid', 'X_test',
        'y_train', 'y_valid', 'y_test',
        
        # Model
        'model', 
        'model_type',
        'selected_model_name',
        'selected_model_timestamp',
        'selected_model_idx',
        'model_metrics',
        'trained_models',
        'best_params',
        'tuning_results',
        
        # SHAP & Explainability
        'explainer', 
        'shap_values',
        'shap_explainer_obj',
        'shap_values_computed',
        'shap_X_explained',
        'shap_feature_importance',
        'shap_expected_value',
        'shap_chat_history',
        'last_ai_analysis',
        'sample_question_selected',
        
        # Prediction
        'prediction_input',
        'prediction_result',
        'feature_stats',
        
        # Feature Engineering configurations
        'missing_config',
        'encoding_config',
        'scaling_config',
        'outlier_config',
        'binning_config',
        'feature_engineering_saved',
        'unlimited_depth',
        
        # Balancing
        'balance_info',
        'balanced_data',
        
        # AI Analysis cache
        'ai_analysis',
        'eda_summary',
        
        # Encoders/Scalers fitted objects
        'label_encoders',
        'onehot_encoder',
        'scaler',
    ]
    
    # Keys that should be reset to empty list
    list_keys = ['selected_features', 'shap_chat_history']
    
    # Keys that should be reset to empty dict
    dict_keys = [
        'model_metrics', 'missing_config', 'encoding_config', 
        'scaling_config', 'outlier_config', 'binning_config',
        'label_encoders', 'feature_stats', 'best_params'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            if key in list_keys:
                st.session_state[key] = []
            elif key in dict_keys:
                st.session_state[key] = {}
            else:
                st.session_state[key] = None
    
    print("DEBUG: Cleared all data-related session state for new dataset")

def get_session_info():
    """Lấy thông tin về session hiện tại"""
    info = {
        "has_data": st.session_state.data is not None,
        "has_processed_data": st.session_state.processed_data is not None,
        "has_model": st.session_state.model is not None or len(st.session_state.get('model_history', [])) > 0,
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

