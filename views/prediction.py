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
        
        # Create input form dynamically based on selected features
        with st.form("prediction_form"):
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
                            
                            # Handle different ranges
                            if max_val - min_val < 10:
                                # Small range - use slider
                                step = 0.1 if (max_val - min_val) < 5 else 1.0
                                input_data[feat] = st.number_input(
                                    feat,
                                    min_value=float(min_val),
                                    max_value=float(max_val) * 1.5,  # Allow slightly above max
                                    value=float(mean_val),
                                    step=step,
                                    key=f"input_{feat}",
                                    help=f"Range: {min_val:.2f} - {max_val:.2f}, Mean: {mean_val:.2f}"
                                )
                            else:
                                # Large range - use number input
                                input_data[feat] = st.number_input(
                                    feat,
                                    min_value=float(min_val) * 0.5 if min_val >= 0 else float(min_val) * 1.5,
                                    max_value=float(max_val) * 1.5,
                                    value=float(mean_val),
                                    step=float((max_val - min_val) / 100),
                                    key=f"input_{feat}",
                                    help=f"Range: {min_val:.2f} - {max_val:.2f}, Mean: {mean_val:.2f}"
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
            
            # Submit button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit_button = st.form_submit_button(
                    "🎯 Dự Đoán Điểm Tín Dụng",
                    use_container_width=True,
                    type="primary"
                )
        
        if submit_button:
            try:
                with st.spinner("Đang dự đoán..."):
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
                    
                    st.success("✅ Đã dự đoán xong! Xem kết quả ở tab 'Kết Quả Dự Đoán'")
                    st.balloons()
                    
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
        
        # Main result display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937 0%, #345f9c 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center;">
                <h2 style="margin: 0; color: white; font-size: 3rem;">{result['credit_score']}</h2>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 1.2rem;">
                    Điểm Tín Dụng
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_color = result['risk_color']
            st.markdown(f"""
            <div style="background-color: #262730; padding: 2rem; border-radius: 15px; 
                        text-align: center; border: 3px solid {risk_color};">
                <h2 style="margin: 0; color: {risk_color}; font-size: 2.5rem;">{result['risk_label_vi']}</h2>
                <p style="margin: 0.5rem 0 0 0; color: #aaa; font-size: 1.2rem;">
                    Mức Độ Rủi Ro
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color: #262730; padding: 2rem; border-radius: 15px; text-align: center;">
                <h2 style="margin: 0; color: white; font-size: 2.5rem;">{result['probability']*100:.1f}%</h2>
                <p style="margin: 0.5rem 0 0 0; color: #aaa; font-size: 1.2rem;">
                    Xác Suất Vỡ Nợ
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Prediction class
        pred_class = result['prediction']
        st.markdown(f"""
        <div style="background-color: {'#2d5016' if pred_class == 0 else '#5c1616'}; 
                    padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: white;">
                {'✅ Khách hàng TỐT - Đủ điều kiện vay' if pred_class == 0 else '⚠️ Khách hàng RỦI RO - Cần xem xét kỹ'}
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Credit score gauge
        st.markdown("#### 📊 Thang Điểm Tín Dụng")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result['credit_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Credit Score", 'font': {'size': 24, 'color': 'white'}},
            delta={'reference': 650, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "lightblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [300, 500], 'color': '#ff4444'},
                    {'range': [500, 650], 'color': '#ffaa00'},
                    {'range': [650, 750], 'color': '#44ff44'},
                    {'range': [750, 850], 'color': '#00ff00'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 650
                }
            }
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            font={'color': "white", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Score interpretation
        st.markdown("#### 📖 Giải Thích Điểm Số")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            interpretation = result['score_interpretation']
            description = result['score_description']
            
            if result['credit_score'] >= 750:
                emoji = "🌟"
                recommendation = "Đủ điều kiện cho các sản phẩm tín dụng với lãi suất ưu đãi"
            elif result['credit_score'] >= 650:
                emoji = "✅"
                recommendation = "Đủ điều kiện cho hầu hết các sản phẩm tín dụng"
            elif result['credit_score'] >= 500:
                emoji = "⚠️"
                recommendation = "Cần xem xét kỹ các điều kiện bổ sung"
            else:
                emoji = "❌"
                recommendation = "Không khuyến nghị phê duyệt hoặc cần tài sản thế chấp"
            
            st.markdown(f"""
            <div style="background-color: #262730; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="margin-top: 0; color: #667eea;">Đánh Giá</h4>
                <p style="margin-bottom: 0.5rem; font-size: 1.1rem;">{emoji} <strong>{interpretation}</strong> - {description}</p>
                <p style="margin-bottom: 0; color: #aaa;">💡 {recommendation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**📊 So Sánh Với Trung Bình**")
            avg_score = 650
            diff = result['credit_score'] - avg_score
            
            st.metric(
                "Điểm trung bình",
                avg_score,
                f"{diff:+d} điểm"
            )
        
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
            
            st.plotly_chart(fig, use_container_width=True)
        
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
            st.dataframe(input_df, use_container_width=True)
    
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
            if st.button("🤖 Tạo Gợi Ý Chi Tiết Từ AI", use_container_width=True, type="primary"):
                with st.spinner("AI đang phân tích và tạo gợi ý..."):
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
            if st.button("📥 Tải Báo Cáo Chi Tiết", use_container_width=True):
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

