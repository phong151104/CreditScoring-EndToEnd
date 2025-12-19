"""
Trang Dự Đoán & Gợi Ý - Prediction & Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.ui_components import show_llm_analysis
from utils.session_state import init_session_state

def render():
    """Render trang dự đoán"""
    init_session_state()
    
    st.markdown("## 🎯 Dự Đoán & Gợi Ý Cải Thiện")
    st.markdown("Nhập thông tin khách hàng để dự đoán điểm tín dụng và nhận gợi ý cải thiện.")
    
    # Check prerequisites
    if st.session_state.model is None:
        st.warning("⚠️ Chưa có mô hình. Vui lòng huấn luyện mô hình trước.")
        return
    
    if st.session_state.selected_features is None or len(st.session_state.selected_features) == 0:
        st.warning("⚠️ Chưa có features được chọn. Vui lòng chọn features trong Feature Engineering.")
        return
    
    # Get the current model name
    current_model_name = st.session_state.get('selected_model_name', st.session_state.get('model_type_select', 'Unknown'))
    st.success(f"✅ Sử dụng mô hình: {current_model_name}")
    
    st.markdown("---")
    
    # Get selected features and their info
    features = st.session_state.selected_features
    
    # Get feature statistics from training data for reference
    train_data = st.session_state.get('train_data')
    feature_stats = {}
    if train_data is not None:
        for feat in features:
            if feat in train_data.columns:
                col_data = train_data[feat]
                if pd.api.types.is_numeric_dtype(col_data):
                    feature_stats[feat] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'dtype': 'numeric'
                    }
                else:
                    feature_stats[feat] = {
                        'unique_values': col_data.unique().tolist(),
                        'dtype': 'categorical'
                    }
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📝 Nhập Thông Tin",
        "🎯 Kết Quả Dự Đoán",
        "💡 Gợi Ý Cải Thiện"
    ])
    
    # Tab 1: Input Form
    with tab1:
        st.markdown("### 📝 Form Nhập Thông Tin Khách Hàng")
        
        st.markdown(f"""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <p style="margin: 0;">📋 Nhập giá trị cho <strong>{len(features)}</strong> đặc trưng đã được chọn để huấn luyện mô hình.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị thông báo nếu vừa dự đoán xong
        if st.session_state.get('_prediction_success'):
            st.success(st.session_state._prediction_success)
            del st.session_state._prediction_success
        
        # Sử dụng form container placeholder để có thể ẩn khi đang xử lý
        form_container = st.container()
        
        with form_container:
            # Create input form dynamically based on selected features
            input_data = {}
            
            # Organize features into columns (3 columns)
            num_cols = 3
            feature_chunks = [features[i:i + num_cols] for i in range(0, len(features), num_cols)]
            
            for chunk in feature_chunks:
                cols = st.columns(num_cols)
                for idx, feat in enumerate(chunk):
                    with cols[idx]:
                        stats = feature_stats.get(feat, {})
                        
                        if stats.get('dtype') == 'numeric':
                            # Numeric input
                            min_val = stats.get('min', 0)
                            max_val = stats.get('max', 1000000)
                            mean_val = stats.get('mean', (min_val + max_val) / 2)
                            
                            # Allow any value - no strict min/max constraints
                            # Just use training data range as guidance
                            if max_val - min_val < 10:
                                # Small range
                                step = 0.1 if (max_val - min_val) < 5 else 1.0
                            else:
                                step = float((max_val - min_val) / 100)
                            
                            # Use text-style number input without strict constraints
                            input_data[feat] = st.number_input(
                                feat,
                                value=float(mean_val),
                                step=step,
                                key=f"input_{feat}",
                                help=f"Training range: {min_val:.2f} - {max_val:.2f}, Mean: {mean_val:.2f} (có thể nhập ngoài khoảng)"
                            )
                        elif stats.get('dtype') == 'categorical':
                            # Categorical input
                            unique_vals = stats.get('unique_values', ['Option 1', 'Option 2'])
                            input_data[feat] = st.selectbox(
                                feat,
                                options=unique_vals,
                                key=f"input_{feat}"
                            )
                        else:
                            # Default to number input if no stats
                            input_data[feat] = st.number_input(
                                feat,
                                value=0.0,
                                key=f"input_{feat}"
                            )
            
            st.markdown("---")
            
            # Submit button - sử dụng placeholder để tránh nhân đôi hoàn toàn
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                button_placeholder = st.empty()
                clicked = button_placeholder.button("🎯 Dự Đoán Điểm Tín Dụng", key="predict_btn", type="primary", width='stretch')
        
        # Xử lý bên ngoài form container
        if clicked:
            # Xóa toàn bộ nút
            button_placeholder.empty()
            
            # Hiển thị spinner ở vị trí riêng
            with st.spinner("Đang dự đoán..."):
                try:
                    # Import prediction backend
                    from backend.models.predictor import predict_single, get_feature_contributions
                    
                    # Make prediction
                    result = predict_single(
                        model=st.session_state.model,
                        input_data=input_data,
                        feature_names=features,
                        feature_stats=feature_stats
                    )
                    
                    # Get feature contributions
                    shap_explainer = st.session_state.get('shap_explainer_obj')
                    contributions = get_feature_contributions(
                        model=st.session_state.model,
                        input_data=input_data,
                        feature_names=features,
                        shap_explainer=shap_explainer
                    )
                    
                    # Store results in session state
                    st.session_state.prediction_input = input_data
                    st.session_state.prediction_result = result
                    st.session_state.prediction_contributions = contributions
                    
                    st.session_state._prediction_success = "✅ Đã dự đoán xong! Xem kết quả ở tab 'Kết Quả Dự Đoán'"
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
                    import traceback
                    with st.expander("Chi tiết lỗi"):
                        st.code(traceback.format_exc())
    
    # Tab 2: Prediction Results
    with tab2:
        st.markdown("### 🎯 Kết Quả Dự Đoán")
        
        if 'prediction_result' not in st.session_state or st.session_state.prediction_result is None:
            st.info("📝 Vui lòng nhập thông tin và dự đoán ở tab 'Nhập Thông Tin' trước.")
            return
        
        result = st.session_state.prediction_result
        contributions = st.session_state.get('prediction_contributions', [])
        
        # Combined Credit Assessment Card - Compact Design
        pred_class = result['prediction']
        score = result['credit_score']
        probability = result['probability'] * 100
        risk_label = result['risk_label_vi']
        
        # Get approval status from backend (combines PD + Score)
        approval_status = result.get('approval_status', 'conditional')
        approval_label = result.get('approval_label_vi', 'Xem xét')
        approval_color = result.get('approval_color', '#f59e0b')
        
        # Determine score color and label (5-tier)
        if score >= 750:
            score_color = "#10b981"
            score_label = "Xuất sắc"
        elif score >= 650:
            score_color = "#22c55e"
            score_label = "Tốt"
        elif score >= 550:
            score_color = "#f59e0b"
            score_label = "Trung bình"
        elif score >= 450:
            score_color = "#ef4444"
            score_label = "Kém"
        else:
            score_color = "#dc2626"
            score_label = "Rất kém"
        
        # Use approval status for header
        if approval_status == 'approved':
            status_bg = "#065f46"
            status_text = "✅ Phê duyệt - Đủ điều kiện vay"
        elif approval_status == 'conditional':
            status_bg = "#92400e"
            status_text = "⚠️ Có thể xem xét vay với điều kiện bổ sung"
        else:
            status_bg = "#991b1b"
            status_text = "❌ Từ chối - Rủi ro cao"
        
        prob_color = result.get('risk_color', '#f59e0b')
        
        # Full-width Credit Assessment Card with embedded gauge
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1f2937 0%, #0f172a 100%); 
                    padding: 1.5rem; border-radius: 16px; 
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3); margin-bottom: 1rem;">
            <div style="background: {status_bg}; padding: 0.6rem 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                <span style="color: white; font-weight: 600; font-size: 1rem;">{status_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Three-column layout inside the card area
        col_score, col_gauge, col_prob = st.columns([1, 1.5, 1])
        
        with col_score:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937 0%, #0f172a 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; height: 200px;
                        display: flex; flex-direction: column; justify-content: center;
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);">
                <div style="font-size: 3.5rem; font-weight: 800; color: {score_color}; line-height: 1;">{score}</div>
                <div style="color: #94a3b8; font-size: 1rem; margin-top: 0.3rem;">điểm tín dụng</div>
                <div style="display: inline-block; background: {score_color}20; color: {score_color}; 
                            padding: 0.4rem 1rem; border-radius: 15px; font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem;">
                    {score_label}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=result['credit_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {
                        'range': [300, 850], 
                        'tickwidth': 1, 
                        'tickcolor': "#475569",
                        'tickfont': {'color': '#94a3b8', 'size': 13},
                        'tickmode': 'array',
                        'tickvals': [300, 500, 650, 750, 850],
                    },
                    'bar': {'color': score_color, 'thickness': 0.3},
                    'bgcolor': "#1e293b",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [300, 500], 'color': 'rgba(239, 68, 68, 0.15)'},
                        {'range': [500, 650], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [650, 750], 'color': 'rgba(34, 197, 94, 0.15)'},
                        {'range': [750, 850], 'color': 'rgba(16, 185, 129, 0.18)'}
                    ],
                    'threshold': {
                        'line': {'color': "#a5b4fc", 'width': 3},
                        'thickness': 0.85,
                        'value': result['credit_score']
                    }
                }
            ))
            
            fig.update_layout(
                template="plotly_dark",
                height=200,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#e2e8f0", 'family': "Inter, Arial, sans-serif"},
                margin=dict(l=20, r=20, t=40, b=10)
            )
            
            st.plotly_chart(fig, width='stretch')
        
        with col_prob:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937 0%, #0f172a 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; height: 200px;
                        display: flex; flex-direction: column; justify-content: center;
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);">
                <div style="font-size: 3rem; font-weight: 700; color: {prob_color}; line-height: 1;">{probability:.1f}%</div>
                <div style="color: #94a3b8; font-size: 1rem; margin-top: 0.3rem;">xác suất vỡ nợ</div>
                <div style="display: inline-block; background: {result['risk_color']}20; color: {result['risk_color']}; 
                            padding: 0.4rem 1rem; border-radius: 15px; font-size: 0.9rem; font-weight: 600; margin-top: 0.5rem;">
                    {risk_label}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Legend row
        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 2.5rem; font-size: 0.9rem; 
                    padding: 1rem; background: rgba(30, 41, 59, 0.5); border-radius: 10px; margin-top: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #ef4444;"></span>
                <span style="color: #94a3b8;">300-500: Kém</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #f59e0b;"></span>
                <span style="color: #94a3b8;">500-650: TB</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #22c55e;"></span>
                <span style="color: #94a3b8;">650-750: Tốt</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="width: 10px; height: 10px; border-radius: 50%; background: #10b981;"></span>
                <span style="color: #94a3b8;">750-850: Xuất sắc</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature contributions
        st.markdown("#### 🔍 Các Yếu Tố Ảnh Hưởng")
        
        if contributions:
            # Sort by absolute impact and take top 10
            sorted_contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)[:10]
            
            feature_names_plot = [c[0] for c in sorted_contributions]
            shap_vals = [c[1] for c in sorted_contributions]
            colors = ['#ff4444' if v > 0 else '#44ff44' for v in shap_vals]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=feature_names_plot,
                x=shap_vals,
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_vals],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Feature Impact on Prediction",
                xaxis_title="Impact (contribution to risk)",
                template="plotly_dark",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px;">
            <p style="margin: 0; font-size: 0.9rem;">
                💡 <strong>Chú thích:</strong><br>
                <span style="color: #44ff44;">●</span> Màu xanh: Yếu tố tác động tích cực (giảm rủi ro)<br>
                <span style="color: #ff4444;">●</span> Màu đỏ: Yếu tố tác động tiêu cực (tăng rủi ro)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show input data summary
        st.markdown("---")
        st.markdown("#### 📋 Thông Tin Đã Nhập")
        
        input_data = st.session_state.get('prediction_input', {})
        if input_data:
            input_df = pd.DataFrame([input_data]).T
            input_df.columns = ['Giá trị']
            st.dataframe(input_df, width='stretch')
    
    # Tab 3: Recommendations
    with tab3:
        st.markdown("### 💡 Gợi Ý Cải Thiện Điểm Tín Dụng")
        
        if 'prediction_result' not in st.session_state or st.session_state.prediction_result is None:
            st.info("📝 Vui lòng nhập thông tin và dự đoán trước.")
            return
        
        result = st.session_state.prediction_result
        input_data = st.session_state.get('prediction_input', {})
        contributions = st.session_state.get('prediction_contributions', [])
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
            <h4 style="margin-top: 0; color: #667eea;">🎯 Mục Tiêu Cải Thiện</h4>
            <p style="margin-bottom: 0;">Dưới đây là các gợi ý cụ thể để nâng cao điểm tín dụng của bạn.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get recommendations from backend
        from backend.models.predictor import generate_recommendations
        recommendations = generate_recommendations(result, input_data, contributions)
        
        # Display recommendations
        if recommendations:
            st.markdown("#### 📈 Các Hành Động Ưu Tiên")
            
            for i, rec in enumerate(recommendations[:5]):
                priority_color = '#ff4444' if rec['priority'] == 'High' else '#ffaa00' if rec['priority'] == 'Medium' else '#44ff44'
                
                st.markdown(f"""
                <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;
                            border-left: 4px solid {priority_color};">
                    <h5 style="margin: 0; color: white;">{i+1}. {rec['feature']}</h5>
                    <p style="margin: 0.5rem 0; color: #aaa;">
                        Giá trị hiện tại: <strong>{rec['current_value']}</strong> | 
                        Mục tiêu: <strong>{rec['target']}</strong>
                    </p>
                    <p style="margin: 0; color: #ccc;">💡 {rec['advice']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # AI-generated recommendations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("---")
            if st.button("✨ Tạo Gợi Ý Chi Tiết Từ AI", width='stretch', type="primary"):
                with st.spinner("✨ AI đang phân tích và tạo gợi ý..."):
                    # Get model name
                    model_name = st.session_state.get('selected_model_name', 'Unknown')
                    
                    # Prepare context for AI
                    top_negative = [(f, c) for f, c in contributions if c > 0][:5]
                    top_positive = [(f, c) for f, c in contributions if c < 0][:5]
                    
                    negative_factors_text = "\n".join([f"- {f}: impact = {c:+.4f}" for f, c in top_negative])
                    positive_factors_text = "\n".join([f"- {f}: impact = {c:+.4f}" for f, c in top_positive])
                    
                    input_summary = "\n".join([f"- {k}: {v}" for k, v in list(input_data.items())[:10]])
                    
                    # Try to use AI
                    try:
                        from backend.llm_integration import create_shap_analyzer, LLMConfig
                        
                        if LLMConfig.GOOGLE_API_KEY:
                            analyzer = create_shap_analyzer()
                            
                            prompt = f"""Phân tích kết quả dự đoán tín dụng và đưa ra gợi ý cải thiện:

**Kết quả dự đoán:**
- Điểm tín dụng: {result['credit_score']}
- Xác suất vỡ nợ: {result['probability']*100:.1f}%
- Mức độ rủi ro: {result['risk_label_vi']}
- Phân loại: {'Rủi ro' if result['prediction'] == 1 else 'Tốt'}

**Thông tin khách hàng:**
{input_summary}

**Yếu tố tăng rủi ro (cần cải thiện):**
{negative_factors_text if negative_factors_text else 'Không có'}

**Yếu tố giảm rủi ro (điểm mạnh):**
{positive_factors_text if positive_factors_text else 'Không có'}

Hãy đưa ra:
1. Phân tích chi tiết về tình trạng tín dụng hiện tại
2. 3-5 gợi ý cụ thể để cải thiện điểm tín dụng (ưu tiên theo tác động)
3. Dự báo cải thiện nếu thực hiện các gợi ý
4. Lưu ý và cảnh báo quan trọng

Trả lời bằng tiếng Việt, sử dụng markdown format."""

                            ai_response = analyzer._call_llm(prompt, 
                                "Bạn là chuyên gia tư vấn tín dụng, phân tích kết quả đánh giá rủi ro và đưa ra gợi ý cải thiện.")
                            
                            show_llm_analysis("Gợi ý cải thiện từ AI", ai_response)
                        else:
                            # Fallback without AI
                            _show_fallback_recommendations(result, input_data, contributions)
                    except Exception as e:
                        st.warning(f"Không thể kết nối AI: {str(e)}")
                        _show_fallback_recommendations(result, input_data, contributions)
        
        with col2:
            st.markdown("#### 🎯 Mục Tiêu")
            
            target_score = st.number_input(
                "Điểm mục tiêu:",
                min_value=result['credit_score'],
                max_value=850,
                value=min(result['credit_score'] + 100, 850),
                step=10
            )
            
            improvement_needed = target_score - result['credit_score']
            
            st.metric(
                "Cần cải thiện",
                f"+{improvement_needed} điểm"
            )
            
            estimated_time = max(3, improvement_needed // 10)
            st.metric(
                "Thời gian ước tính",
                f"~{estimated_time} tháng"
            )
        
        st.markdown("---")
        
        # Action plan
        st.markdown("#### 📅 Kế Hoạch Hành Động")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1f2937 0%, #345f9c 100%); 
                        padding: 1.5rem; border-radius: 10px;">
                <h3 style="margin: 0; color: white;">Tháng 1-2</h3>
                <ul style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
                    <li>Giảm credit utilization</li>
                    <li>Thiết lập auto-payment</li>
                    <li>Review credit report</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #345f9c 0%, #1f2937 100%); 
                        padding: 1.5rem; border-radius: 10px;">
                <h3 style="margin: 0; color: white;">Tháng 3-4</h3>
                <ul style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
                    <li>Trả bớt các khoản nợ</li>
                    <li>Không mở tài khoản mới</li>
                    <li>Duy trì payment history</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1f2937 0%, #345f9c 100%); 
                        padding: 1.5rem; border-radius: 10px;">
                <h3 style="margin: 0; color: white;">Tháng 5-6</h3>
                <ul style="color: rgba(255,255,255,0.9); margin-bottom: 0;">
                    <li>Kiểm tra tiến độ</li>
                    <li>Điều chỉnh chiến lược</li>
                    <li>Đánh giá lại điểm số</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Download report
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("📥 Tải Báo Cáo Chi Tiết", width='stretch'):
                # Generate report content
                report_content = f"""
# BÁO CÁO ĐÁNH GIÁ TÍN DỤNG

## Thông Tin Chung
- Ngày đánh giá: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
- Mô hình sử dụng: {current_model_name}

## Kết Quả Đánh Giá
- Điểm tín dụng: {result['credit_score']}/850
- Xác suất rủi ro: {result['probability']*100:.1f}%
- Mức độ rủi ro: {result['risk_label_vi']}
- Phân loại: {'Rủi ro cao' if result['prediction'] == 1 else 'Tín dụng tốt'}

## Giải Thích
- Đánh giá: {result['score_interpretation']}
- Mô tả: {result['score_description']}

## Thông Tin Khách Hàng
{chr(10).join([f'- {k}: {v}' for k, v in input_data.items()])}

## Các Yếu Tố Ảnh Hưởng
{chr(10).join([f'- {f}: {c:+.4f}' for f, c in contributions[:10]])}
"""
                st.download_button(
                    "📄 Tải xuống (.txt)",
                    report_content,
                    file_name="credit_report.txt",
                    mime="text/plain"
                )
                st.success("✅ Đã tạo báo cáo!")


def _show_fallback_recommendations(result, input_data, contributions):
    """Show fallback recommendations when AI is not available"""
    
    top_negative = [(f, c) for f, c in contributions if c > 0][:3]
    
    fallback_response = f"""
## 🎯 Phân Tích Tình Huống Hiện Tại

Điểm tín dụng hiện tại của bạn là **{result['credit_score']}** điểm, thuộc nhóm **{result['risk_label_vi']}**.

### 📈 Các Hành Động Ưu Tiên

"""
    
    for i, (feat, impact) in enumerate(top_negative):
        fallback_response += f"""
**{i+1}. Cải thiện {feat}**
- Giá trị hiện tại: {input_data.get(feat, 'N/A')}
- Tác động: {impact:+.4f}
- Gợi ý: Giảm giá trị này để cải thiện điểm tín dụng

"""
    
    fallback_response += f"""
### 📊 Dự Báo Cải Thiện

Nếu thực hiện các gợi ý trên trong 6 tháng, điểm tín dụng có thể tăng **{np.random.randint(30, 80)}** điểm.

### ⚠️ Lưu Ý

*Đây là gợi ý tự động. Để có phân tích chi tiết hơn từ AI, vui lòng cấu hình GOOGLE_API_KEY trong file .env*
"""
    
    show_llm_analysis("Gợi ý cải thiện", fallback_response)

