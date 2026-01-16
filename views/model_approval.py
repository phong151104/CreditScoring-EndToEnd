"""
Model Approval Page - For Validators to review and approve/reject models
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from utils.session_state import init_session_state
from utils.permissions import require_role, has_permission, get_current_user


def render():
    """Render model approval page"""
    init_session_state()
    
    st.markdown("## ✅ Phê duyệt & Đánh giá Mô hình")
    st.markdown("Xem xét, đánh giá và phê duyệt các mô hình đã được huấn luyện.")
    
    # Initialize approval history in session state
    if 'model_approvals' not in st.session_state:
        st.session_state.model_approvals = []
    
    # Check if there are trained models
    model_history = st.session_state.get('model_history', [])
    current_model = st.session_state.get('model')
    
    if not model_history and current_model is None:
        st.warning("⚠️ Chưa có mô hình nào được huấn luyện. Vui lòng chờ Admin huấn luyện mô hình.")
        
        # Show any previous approval history
        if st.session_state.model_approvals:
            st.markdown("---")
            st.markdown("### 📋 Lịch sử Phê duyệt")
            _show_approval_history()
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Mô hình Chờ duyệt",
        "📝 Đánh giá & Nhận xét",
        "📋 Lịch sử Phê duyệt"
    ])
    
    with tab1:
        _render_pending_models(model_history, current_model)
    
    with tab2:
        _render_evaluation_form()
    
    with tab3:
        _show_approval_history()


def _render_pending_models(model_history, current_model):
    """Render list of models pending approval"""
    st.markdown("### 📊 Danh sách Mô hình")
    
    if model_history:
        # Create dataframe from model history
        models_data = []
        for idx, model_info in enumerate(model_history):
            status = _get_approval_status(model_info.get('timestamp', ''))
            models_data.append({
                'STT': idx + 1,
                'Model': model_info.get('model_type', 'Unknown'),
                'Timestamp': model_info.get('timestamp', 'N/A'),
                'AUC': f"{model_info.get('auc', 0):.4f}",
                'Accuracy': f"{model_info.get('accuracy', 0):.4f}",
                'F1': f"{model_info.get('f1', 0):.4f}",
                'Trạng thái': status
            })
        
        df = pd.DataFrame(models_data)
        
        # Style the dataframe
        st.dataframe(df, width='stretch', hide_index=True)
        
    elif current_model is not None:
        # Show current model info
        metrics = st.session_state.get('model_metrics', {})
        model_name = st.session_state.get('selected_model_name', 'Unknown')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", model_name)
        with col2:
            st.metric("AUC", f"{metrics.get('auc', 0):.4f}")
        with col3:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        with col4:
            st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
        
        # Show status
        status = _get_approval_status(st.session_state.get('selected_model_timestamp', ''))
        
        if status == "✅ Đã duyệt":
            st.success(f"Trạng thái: {status}")
        elif status == "❌ Từ chối":
            st.error(f"Trạng thái: {status}")
        else:
            st.info(f"Trạng thái: {status}")
    
    # Show SHAP summary if available
    st.markdown("---")
    st.markdown("#### 🔍 Tóm tắt SHAP")
    
    if st.session_state.get('shap_feature_importance') is not None:
        importance_df = st.session_state.shap_feature_importance
        if isinstance(importance_df, pd.DataFrame):
            st.dataframe(importance_df.head(10), width='stretch')
        else:
            st.info("👁️ Bấm vào 'Model Explanation' để xem chi tiết SHAP values")
    else:
        st.info("⚠️ Chưa có SHAP analysis. Admin cần chạy SHAP trước.")


def _render_evaluation_form():
    """Render evaluation and comment form"""
    st.markdown("### 📝 Form Đánh giá Mô hình")
    
    current_user = get_current_user()
    
    # Check if there's a model to evaluate
    if st.session_state.get('model') is None:
        st.warning("⚠️ Không có mô hình để đánh giá.")
        return
    
    model_name = st.session_state.get('selected_model_name', 'Unknown')
    model_timestamp = st.session_state.get('selected_model_timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <h4 style='margin: 0; color: #667eea;'>Mô hình: {model_name}</h4>
        <p style='margin: 0.5rem 0 0 0; color: #94a3b8;'>Timestamp: {model_timestamp}</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("evaluation_form"):
        # Rating criteria
        st.markdown("#### 📊 Tiêu chí Đánh giá")
        
        col1, col2 = st.columns(2)
        
        with col1:
            performance_rating = st.select_slider(
                "Hiệu suất mô hình",
                options=["Kém", "Trung bình", "Khá", "Tốt", "Xuất sắc"],
                value="Khá"
            )
            
            robustness_rating = st.select_slider(
                "Độ ổn định/Robust",
                options=["Kém", "Trung bình", "Khá", "Tốt", "Xuất sắc"],
                value="Khá"
            )
        
        with col2:
            interpretability_rating = st.select_slider(
                "Khả năng giải thích",
                options=["Kém", "Trung bình", "Khá", "Tốt", "Xuất sắc"],
                value="Khá"
            )
            
            compliance_rating = st.select_slider(
                "Tuân thủ quy định",
                options=["Kém", "Trung bình", "Khá", "Tốt", "Xuất sắc"],
                value="Khá"
            )
        
        # Comments
        st.markdown("#### 💬 Nhận xét")
        
        comments = st.text_area(
            "Nhận xét chi tiết",
            placeholder="Nhập nhận xét của bạn về mô hình...",
            height=150
        )
        
        concerns = st.text_area(
            "Rủi ro/Khuyến nghị (nếu có)",
            placeholder="Các rủi ro khi sử dụng mô hình hoặc khuyến nghị cải thiện...",
            height=100
        )
        
        # Decision
        st.markdown("#### ✅ Quyết định")
        
        decision = st.radio(
            "Phê duyệt mô hình",
            options=["Phê duyệt", "Từ chối", "Yêu cầu chỉnh sửa"],
            horizontal=True
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            submitted = st.form_submit_button("📤 Gửi Đánh giá", type="primary", width='stretch')
        
        if submitted:
            # Save approval record
            approval_record = {
                'model_name': model_name,
                'model_timestamp': model_timestamp,
                'evaluator': current_user.display_name if current_user else 'Unknown',
                'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ratings': {
                    'performance': performance_rating,
                    'robustness': robustness_rating,
                    'interpretability': interpretability_rating,
                    'compliance': compliance_rating
                },
                'comments': comments,
                'concerns': concerns,
                'decision': decision
            }
            
            st.session_state.model_approvals.append(approval_record)
            
            if decision == "Phê duyệt":
                st.success("✅ Mô hình đã được PHÊ DUYỆT!")
            elif decision == "Từ chối":
                st.error("❌ Mô hình đã bị TỪ CHỐI.")
            else:
                st.warning("⚠️ Đã yêu cầu Admin chỉnh sửa mô hình.")


def _show_approval_history():
    """Show history of model approvals"""
    st.markdown("### 📋 Lịch sử Phê duyệt")
    
    approvals = st.session_state.get('model_approvals', [])
    
    if not approvals:
        st.info("📭 Chưa có lịch sử phê duyệt nào.")
        return
    
    # Show in reverse chronological order
    for idx, record in enumerate(reversed(approvals)):
        decision = record.get('decision', 'Unknown')
        
        if decision == "Phê duyệt":
            icon = "✅"
            color = "#10b981"
        elif decision == "Từ chối":
            icon = "❌"
            color = "#ef4444"
        else:
            icon = "⚠️"
            color = "#f59e0b"
        
        with st.expander(f"{icon} {record.get('model_name', 'Unknown')} - {record.get('evaluation_time', '')}"):
            st.markdown(f"""
            **Người đánh giá:** {record.get('evaluator', 'Unknown')}
            
            **Đánh giá:**
            - Hiệu suất: {record.get('ratings', {}).get('performance', 'N/A')}
            - Độ ổn định: {record.get('ratings', {}).get('robustness', 'N/A')}
            - Khả năng giải thích: {record.get('ratings', {}).get('interpretability', 'N/A')}
            - Tuân thủ: {record.get('ratings', {}).get('compliance', 'N/A')}
            
            **Nhận xét:** {record.get('comments', 'Không có')}
            
            **Rủi ro/Khuyến nghị:** {record.get('concerns', 'Không có')}
            
            **Quyết định:** <span style="color: {color}; font-weight: bold;">{decision}</span>
            """, unsafe_allow_html=True)


def _get_approval_status(model_timestamp):
    """Get approval status for a model based on timestamp"""
    approvals = st.session_state.get('model_approvals', [])
    
    for record in reversed(approvals):
        if record.get('model_timestamp') == model_timestamp:
            decision = record.get('decision', '')
            if decision == "Phê duyệt":
                return "✅ Đã duyệt"
            elif decision == "Từ chối":
                return "❌ Từ chối"
            else:
                return "⚠️ Yêu cầu sửa"
    
    return "⏳ Chờ duyệt"
