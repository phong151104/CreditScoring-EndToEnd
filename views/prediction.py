"""
Trang Dự Đoán & Gợi Ý - Prediction & Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.ui_components import show_llm_analysis, show_processing_placeholder
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
    
    st.success(f"✅ Sử dụng mô hình: {st.session_state.model_type}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📝 Nhập Thông Tin",
        "🎯 Kết Quả Dự Đoán",
        "💡 Gợi Ý Cải Thiện"
    ])
    
    # Tab 1: Input Form
    with tab1:
        st.markdown("### 📝 Form Nhập Thông Tin Khách Hàng")
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
            <p style="margin: 0;">📋 Vui lòng điền đầy đủ thông tin để nhận dự đoán chính xác nhất.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create input form based on selected features
        with st.form("prediction_form"):
            st.markdown("#### 👤 Thông Tin Cá Nhân")
            
            col1, col2, col3 = st.columns(3)
            
            # Mock input fields (would be dynamic based on selected features)
            input_data = {}
            
            with col1:
                input_data['age'] = st.number_input(
                    "Tuổi",
                    min_value=18,
                    max_value=100,
                    value=35,
                    step=1
                )
                
                input_data['income'] = st.number_input(
                    "Thu nhập hàng tháng (VNĐ)",
                    min_value=0,
                    max_value=1000000000,
                    value=15000000,
                    step=1000000,
                    format="%d"
                )
                
                input_data['employment_years'] = st.number_input(
                    "Số năm làm việc",
                    min_value=0,
                    max_value=50,
                    value=5,
                    step=1
                )
            
            with col2:
                input_data['loan_amount'] = st.number_input(
                    "Số tiền vay (VNĐ)",
                    min_value=0,
                    max_value=5000000000,
                    value=100000000,
                    step=10000000,
                    format="%d"
                )
                
                input_data['existing_loans'] = st.number_input(
                    "Số khoản vay hiện tại",
                    min_value=0,
                    max_value=10,
                    value=1,
                    step=1
                )
                
                input_data['monthly_debt'] = st.number_input(
                    "Tổng nợ hàng tháng (VNĐ)",
                    min_value=0,
                    max_value=100000000,
                    value=5000000,
                    step=1000000,
                    format="%d"
                )
            
            with col3:
                input_data['credit_history'] = st.selectbox(
                    "Lịch sử tín dụng",
                    ["Excellent", "Good", "Fair", "Poor", "No History"]
                )
                
                input_data['education'] = st.selectbox(
                    "Trình độ học vấn",
                    ["Postgraduate", "Graduate", "High School", "Other"]
                )
                
                input_data['marital_status'] = st.selectbox(
                    "Tình trạng hôn nhân",
                    ["Single", "Married", "Divorced", "Widowed"]
                )
            
            st.markdown("---")
            st.markdown("#### 🏠 Thông Tin Bổ Sung")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                input_data['home_ownership'] = st.selectbox(
                    "Tình trạng nhà ở",
                    ["Own", "Rent", "Mortgage", "Other"]
                )
                
                input_data['dependents'] = st.number_input(
                    "Số người phụ thuộc",
                    min_value=0,
                    max_value=10,
                    value=2,
                    step=1
                )
            
            with col2:
                input_data['bank_account_years'] = st.number_input(
                    "Số năm có tài khoản ngân hàng",
                    min_value=0,
                    max_value=50,
                    value=10,
                    step=1
                )
                
                input_data['credit_cards'] = st.number_input(
                    "Số thẻ tín dụng",
                    min_value=0,
                    max_value=10,
                    value=2,
                    step=1
                )
            
            with col3:
                input_data['late_payments'] = st.number_input(
                    "Số lần trả nợ muộn (12 tháng)",
                    min_value=0,
                    max_value=50,
                    value=0,
                    step=1
                )
                
                input_data['credit_utilization'] = st.slider(
                    "Tỷ lệ sử dụng tín dụng (%)",
                    0, 100, 30,
                    help="Tỷ lệ tín dụng đã sử dụng / tổng hạn mức"
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
            # Store input data in session state
            st.session_state.prediction_input = input_data
            
            # Mock prediction
            pred_proba = np.random.uniform(0.2, 0.95)
            credit_score = int(300 + pred_proba * 550)  # Scale 300-850
            
            st.session_state.prediction_result = {
                'probability': pred_proba,
                'credit_score': credit_score,
                'risk_level': 'Low' if pred_proba < 0.3 else 'Medium' if pred_proba < 0.6 else 'High'
            }
            
            st.success("✅ Đã tính toán xong! Xem kết quả ở tab 'Kết Quả Dự Đoán'")
            st.balloons()
    
    # Tab 2: Prediction Results
    with tab2:
        st.markdown("### 🎯 Kết Quả Dự Đoán")
        
        if 'prediction_result' not in st.session_state:
            st.info("📝 Vui lòng nhập thông tin và dự đoán ở tab 'Nhập Thông Tin' trước.")
            return
        
        result = st.session_state.prediction_result
        
        # Main result display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center;">
                <h2 style="margin: 0; color: white; font-size: 3rem;">{result['credit_score']}</h2>
                <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9); font-size: 1.2rem;">
                    Điểm Tín Dụng
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_color = '#44ff44' if result['risk_level'] == 'Low' else '#ffaa00' if result['risk_level'] == 'Medium' else '#ff4444'
            st.markdown(f"""
            <div style="background-color: #262730; padding: 2rem; border-radius: 15px; 
                        text-align: center; border: 3px solid {risk_color};">
                <h2 style="margin: 0; color: {risk_color}; font-size: 2.5rem;">{result['risk_level']}</h2>
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
            if result['credit_score'] >= 750:
                interpretation = "🌟 **Xuất sắc** - Khách hàng có tín dụng rất tốt, rủi ro thấp"
                recommendation = "Đủ điều kiện cho các sản phẩm tín dụng với lãi suất ưu đãi"
            elif result['credit_score'] >= 650:
                interpretation = "✅ **Tốt** - Khách hàng có tín dụng tốt, rủi ro trung bình thấp"
                recommendation = "Đủ điều kiện cho hầu hết các sản phẩm tín dụng"
            elif result['credit_score'] >= 500:
                interpretation = "⚠️ **Trung bình** - Khách hàng cần cải thiện tín dụng"
                recommendation = "Cần xem xét kỹ các điều kiện bổ sung"
            else:
                interpretation = "❌ **Kém** - Khách hàng có rủi ro cao"
                recommendation = "Không khuyến nghị phê duyệt hoặc cần tài sản thế chấp"
            
            st.markdown(f"""
            <div style="background-color: #262730; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="margin-top: 0; color: #667eea;">Đánh Giá</h4>
                <p style="margin-bottom: 0.5rem; font-size: 1.1rem;">{interpretation}</p>
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
        
        # SHAP explanation for this prediction
        st.markdown("#### 🔍 Các Yếu Tố Ảnh Hưởng")
        
        # Mock SHAP values for this prediction
        factors = [
            ('Income', np.random.uniform(-0.3, 0.3)),
            ('Loan Amount', np.random.uniform(-0.3, 0.3)),
            ('Credit History', np.random.uniform(-0.3, 0.3)),
            ('Late Payments', np.random.uniform(-0.3, 0.3)),
            ('Employment Years', np.random.uniform(-0.3, 0.3)),
            ('Credit Utilization', np.random.uniform(-0.3, 0.3)),
            ('Existing Loans', np.random.uniform(-0.3, 0.3)),
            ('Monthly Debt', np.random.uniform(-0.3, 0.3)),
        ]
        
        # Sort by absolute impact
        factors.sort(key=lambda x: abs(x[1]), reverse=True)
        
        feature_names = [f[0] for f in factors]
        shap_vals = [f[1] for f in factors]
        colors = ['#ff4444' if v < 0 else '#44ff44' for v in shap_vals]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=feature_names,
            x=shap_vals,
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.3f}" for v in shap_vals],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Feature Impact on Prediction",
            xaxis_title="Impact (SHAP value)",
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
    
    # Tab 3: Recommendations
    with tab3:
        st.markdown("### 💡 Gợi Ý Cải Thiện Điểm Tín Dụng")
        
        if 'prediction_result' not in st.session_state:
            st.info("📝 Vui lòng nhập thông tin và dự đoán trước.")
            return
        
        result = st.session_state.prediction_result
        input_data = st.session_state.get('prediction_input', {})
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
            <h4 style="margin-top: 0; color: #667eea;">🎯 Mục Tiêu Cải Thiện</h4>
            <p style="margin-bottom: 0;">Dưới đây là các gợi ý cụ thể để nâng cao điểm tín dụng của bạn.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AI-generated recommendations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🤖 Tạo Gợi Ý Từ AI", use_container_width=True, type="primary"):
                with st.spinner("AI đang phân tích và tạo gợi ý..."):
                    
                    ai_recommendations = f"""
                    **🎯 Phân Tích Tình Huống Hiện Tại**
                    
                    Điểm tín dụng hiện tại của bạn là **{result['credit_score']}** điểm, thuộc nhóm 
                    **{result['risk_level']} risk**. Dựa trên phân tích chi tiết, đây là các gợi ý cải thiện:
                    
                    **📈 Các Hành Động Ưu Tiên (Tác động cao)**
                    
                    1. **Giảm tỷ lệ sử dụng tín dụng**
                       - Hiện tại: {input_data.get('credit_utilization', 30)}%
                       - Mục tiêu: < 30%
                       - Tác động: +{np.random.randint(20, 40)} điểm
                       - Cách thực hiện: Trả bớt nợ thẻ tín dụng hoặc tăng hạn mức
                    
                    2. **Cải thiện tỷ lệ thu nhập/nợ**
                       - Hiện tại: {(input_data.get('monthly_debt', 5000000) / input_data.get('income', 15000000) * 100):.1f}%
                       - Mục tiêu: < 30%
                       - Tác động: +{np.random.randint(15, 30)} điểm
                       - Cách thực hiện: Tăng thu nhập hoặc giảm các khoản nợ định kỳ
                    
                    3. **Đảm bảo thanh toán đúng hạn**
                       - Số lần trả muộn: {input_data.get('late_payments', 0)}
                       - Mục tiêu: 0 lần trả muộn trong 12 tháng
                       - Tác động: +{np.random.randint(10, 25)} điểm
                       - Cách thực hiện: Thiết lập thanh toán tự động
                    
                    **⏱️ Các Hành Động Dài Hạn**
                    
                    4. **Duy trì lịch sử tín dụng lâu dài**
                       - Không đóng các tài khoản cũ
                       - Tác động: +{np.random.randint(5, 15)} điểm trong 1 năm
                    
                    5. **Đa dạng hóa các loại tín dụng**
                       - Cân nhắc có cả tín dụng xoay vòng (thẻ) và tín dụng trả góp (vay)
                       - Tác động: +{np.random.randint(5, 15)} điểm
                    
                    **📊 Dự Báo Cải Thiện**
                    
                    Nếu thực hiện đầy đủ các gợi ý trên trong 6 tháng, điểm tín dụng của bạn có thể 
                    tăng lên **{result['credit_score'] + np.random.randint(50, 100)} điểm** 
                    (tăng {np.random.randint(50, 100)} điểm).
                    
                    **💰 Lợi Ích Khi Cải Thiện**
                    
                    - Lãi suất vay giảm: {np.random.uniform(1, 3):.1f}% → Tiết kiệm hàng triệu đồng
                    - Dễ dàng được phê duyệt các sản phẩm tín dụng
                    - Hạn mức tín dụng cao hơn
                    - Điều kiện vay tốt hơn
                    
                    ⚡ *Đây là gợi ý mô phỏng. Backend sẽ tích hợp LLM để phân tích chi tiết hơn.*
                    """
                    
                    show_llm_analysis("Gợi ý cải thiện điểm tín dụng", ai_recommendations)
        
        with col2:
            st.markdown("#### 🎯 Mục Tiêu")
            
            target_score = st.number_input(
                "Điểm mục tiêu:",
                result['credit_score'],
                850,
                min(result['credit_score'] + 100, 850),
                10
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
            
            st.markdown("**⚙️ Tùy Chọn**")
            
            show_detailed = st.checkbox("Hiện chi tiết", value=True)
            include_examples = st.checkbox("Bao gồm ví dụ", value=True)
        
        st.markdown("---")
        
        # Action plan
        st.markdown("#### 📅 Kế Hoạch Hành Động")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
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
            <div style="background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); 
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
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
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
                show_processing_placeholder("Tạo báo cáo PDF với tất cả thông tin và gợi ý")
                st.success("✅ Đã tạo báo cáo!")

