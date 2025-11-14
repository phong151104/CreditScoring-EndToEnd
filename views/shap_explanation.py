"""
Trang Giải Thích SHAP - Model Explanation with SHAP
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.ui_components import show_llm_analysis, show_processing_placeholder
from utils.session_state import init_session_state

def render():
    """Render trang giải thích SHAP"""
    init_session_state()
    
    st.markdown("## 🔍 Giải Thích Mô Hình Với SHAP")
    st.markdown("Hiểu rõ cách mô hình đưa ra quyết định thông qua SHAP (SHapley Additive exPlanations).")
    
    # Check prerequisites
    if st.session_state.data is None:
        st.warning("⚠️ Chưa có dữ liệu. Vui lòng upload dữ liệu trước.")
        return
    
    if st.session_state.model is None:
        st.warning("⚠️ Chưa có mô hình. Vui lòng huấn luyện mô hình trước.")
        return
    
    st.success(f"✅ Đang phân tích mô hình: {st.session_state.model_type}")
    
    st.markdown("---")
    
    # SHAP explainer initialization
    if st.session_state.explainer is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Khởi Tạo SHAP Explainer", use_container_width=True, type="primary"):
                with st.spinner("Đang khởi tạo SHAP explainer..."):
                    show_processing_placeholder("Tạo SHAP explainer cho mô hình")
                    st.session_state.explainer = "initialized"
                    st.session_state.shap_values = "computed"
                    st.success("✅ Đã khởi tạo SHAP explainer!")
                    st.rerun()
        
        st.info("💡 Nhấn nút trên để tính toán SHAP values cho mô hình")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🌍 Global Explanation",
        "🎯 Local Explanation",
        "🤖 AI Interpretation"
    ])
    
    # Tab 1: Global Explanation
    with tab1:
        st.markdown("### 🌍 Global Feature Importance")
        st.markdown("Mức độ quan trọng tổng thể của các đặc trưng đối với mô hình.")
        
        # Mock feature importance data
        features = st.session_state.selected_features[:15] if len(st.session_state.selected_features) >= 15 else st.session_state.selected_features
        importance_values = np.random.random(len(features))
        importance_values = np.sort(importance_values)[::-1]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📊 SHAP Summary Plot")
            
            # Create summary plot (bar chart)
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=features,
                x=importance_values,
                orientation='h',
                marker=dict(
                    color=importance_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Impact")
                ),
                text=[f"{val:.3f}" for val in importance_values],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance - Global",
                xaxis_title="Mean |SHAP value|",
                yaxis_title="Features",
                template="plotly_dark",
                height=max(400, len(features) * 30),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Giải thích**: Biểu đồ này cho thấy mức độ ảnh hưởng trung bình của mỗi đặc trưng đến dự đoán của mô hình.")
        
        with col2:
            st.markdown("#### 📋 Top Features")
            
            # Top features table
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance_values
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(
                importance_df.style.format({'Importance': '{:.4f}'})
                .background_gradient(subset=['Importance'], cmap='Reds'),
                use_container_width=True,
                height=400
            )
            
            # Download
            csv = importance_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Tải SHAP Values",
                csv,
                "shap_importance.csv",
                "text/csv"
            )
        
        st.markdown("---")
        
        # SHAP Beeswarm/Violin plot simulation
        st.markdown("#### 🎻 SHAP Value Distribution")
        
        selected_feature = st.selectbox(
            "Chọn đặc trưng để xem phân phối:",
            features,
            key="global_feature_select"
        )
        
        # Mock SHAP value distribution
        shap_values_dist = np.random.randn(200) * np.random.uniform(0.5, 2.0)
        feature_values = np.random.randn(200) * 10 + 50
        
        fig = go.Figure()
        
        # Scatter plot with color based on feature value
        fig.add_trace(go.Scatter(
            x=shap_values_dist,
            y=np.random.randn(200) * 0.1,
            mode='markers',
            marker=dict(
                size=8,
                color=feature_values,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title=selected_feature),
                line=dict(width=0.5, color='white')
            ),
            name=selected_feature,
            text=[f"Value: {v:.2f}<br>SHAP: {s:.3f}" for v, s in zip(feature_values, shap_values_dist)],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"SHAP Value Distribution - {selected_feature}",
            xaxis_title="SHAP value (impact on model output)",
            yaxis_title="",
            template="plotly_dark",
            height=400,
            showlegend=False,
            yaxis=dict(showticklabels=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px;">
            <p style="margin: 0; font-size: 0.9rem;">
                💡 <strong>Cách đọc biểu đồ:</strong><br>
                • Trục X: Giá trị SHAP (dương = tăng xác suất, âm = giảm xác suất)<br>
                • Màu sắc: Giá trị của đặc trưng (đỏ = cao, xanh = thấp)<br>
                • Mật độ điểm: Độ tập trung của các mẫu có giá trị SHAP tương tự
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tab 2: Local Explanation
    with tab2:
        st.markdown("### 🎯 Local Explanation - Giải Thích Từng Mẫu")
        st.markdown("Phân tích chi tiết các yếu tố ảnh hưởng đến dự đoán của một mẫu cụ thể.")
        
        # Sample selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("#### 📋 Chọn Mẫu")
            
            sample_selection_method = st.radio(
                "Phương pháp chọn:",
                ["Chọn theo index", "Chọn ngẫu nhiên", "Nhập dữ liệu mới"],
                key="sample_method"
            )
            
            if sample_selection_method == "Chọn theo index":
                sample_idx = st.number_input(
                    "Index mẫu:",
                    0, len(st.session_state.data) - 1, 0,
                    key="sample_idx"
                )
            elif sample_selection_method == "Chọn ngẫu nhiên":
                if st.button("🎲 Chọn Ngẫu Nhiên", key="random_sample"):
                    sample_idx = np.random.randint(0, len(st.session_state.data))
                    st.session_state.current_sample_idx = sample_idx
                sample_idx = st.session_state.get('current_sample_idx', 0)
            else:
                st.info("📝 Nhập dữ liệu mới ở phần dưới")
                sample_idx = 0
            
            st.markdown(f"**Mẫu đang xem: #{sample_idx}**")
            
            # Prediction info
            pred_proba = np.random.uniform(0.3, 0.9)
            pred_class = 1 if pred_proba > 0.5 else 0
            
            st.markdown("---")
            st.markdown("#### 🎯 Dự Đoán")
            
            st.metric("Xác suất", f"{pred_proba:.1%}")
            st.metric("Phân loại", "✅ Good" if pred_class == 0 else "⚠️ Risk")
        
        with col2:
            st.markdown("#### 💧 SHAP Waterfall Plot")
            
            # Mock SHAP values for single sample
            base_value = 0.5
            shap_values_local = np.random.randn(len(features)) * 0.1
            shap_values_local = np.sort(shap_values_local)
            
            # Create waterfall plot
            cumsum = np.concatenate([[base_value], base_value + np.cumsum(shap_values_local)])
            
            fig = go.Figure()
            
            # Base value
            fig.add_trace(go.Bar(
                name='Base value',
                x=['Base'],
                y=[base_value],
                marker_color='lightgray',
                text=[f"{base_value:.3f}"],
                textposition='outside'
            ))
            
            # Feature contributions
            colors = ['red' if v < 0 else 'green' for v in shap_values_local]
            
            for i, (feat, val) in enumerate(zip(features[:10], shap_values_local[:10])):
                fig.add_trace(go.Bar(
                    name=feat,
                    x=[feat],
                    y=[abs(val)],
                    base=[cumsum[i] if val > 0 else cumsum[i] - abs(val)],
                    marker_color=colors[i],
                    text=[f"{val:+.3f}"],
                    textposition='outside'
                ))
            
            # Final prediction
            fig.add_trace(go.Bar(
                name='Prediction',
                x=['Prediction'],
                y=[cumsum[-1]],
                marker_color='blue',
                text=[f"{cumsum[-1]:.3f}"],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"SHAP Waterfall - Sample #{sample_idx}",
                xaxis_title="Features",
                yaxis_title="Model Output",
                template="plotly_dark",
                height=500,
                showlegend=False,
                barmode='stack'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Feature values for selected sample
        st.markdown("#### 📊 Giá Trị Đặc Trưng Của Mẫu")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top positive impacts
            st.markdown("##### ⬆️ Top Tác Động Tích Cực")
            
            positive_impacts = []
            for i, (feat, shap_val) in enumerate(zip(features, shap_values_local)):
                if shap_val > 0:
                    positive_impacts.append({
                        'Feature': feat,
                        'SHAP Value': shap_val,
                        'Feature Value': np.random.uniform(10, 100)
                    })
            
            if positive_impacts:
                pos_df = pd.DataFrame(positive_impacts).sort_values('SHAP Value', ascending=False).head(5)
                st.dataframe(
                    pos_df.style.format({
                        'SHAP Value': '{:+.4f}',
                        'Feature Value': '{:.2f}'
                    }).background_gradient(subset=['SHAP Value'], cmap='Greens'),
                    use_container_width=True
                )
            else:
                st.info("Không có tác động tích cực")
        
        with col2:
            # Top negative impacts
            st.markdown("##### ⬇️ Top Tác Động Tiêu Cực")
            
            negative_impacts = []
            for i, (feat, shap_val) in enumerate(zip(features, shap_values_local)):
                if shap_val < 0:
                    negative_impacts.append({
                        'Feature': feat,
                        'SHAP Value': shap_val,
                        'Feature Value': np.random.uniform(10, 100)
                    })
            
            if negative_impacts:
                neg_df = pd.DataFrame(negative_impacts).sort_values('SHAP Value').head(5)
                st.dataframe(
                    neg_df.style.format({
                        'SHAP Value': '{:+.4f}',
                        'Feature Value': '{:.2f}'
                    }).background_gradient(subset=['SHAP Value'], cmap='Reds'),
                    use_container_width=True
                )
            else:
                st.info("Không có tác động tiêu cực")
        
        # Force plot alternative
        st.markdown("---")
        st.markdown("#### 🎨 SHAP Force Plot")
        
        # Create force plot visualization
        sorted_indices = np.argsort(np.abs(shap_values_local))[::-1][:10]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_shap = [shap_values_local[i] for i in sorted_indices]
        
        fig = go.Figure()
        
        colors = ['#ff4444' if v < 0 else '#44ff44' for v in sorted_shap]
        
        fig.add_trace(go.Bar(
            x=sorted_shap,
            y=sorted_features,
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.3f}" for v in sorted_shap],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Force Plot - Top Contributing Features",
            xaxis_title="SHAP value (impact on prediction)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: AI Interpretation
    with tab3:
        st.markdown("### 🤖 Giải Thích Bằng AI")
        st.markdown("Phân tích và diễn giải kết quả SHAP bằng ngôn ngữ tự nhiên.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 💬 Phân Tích Tự Động")
            
            analysis_type = st.radio(
                "Loại phân tích:",
                ["Global - Tổng quan mô hình", "Local - Giải thích mẫu cụ thể"],
                key="analysis_type"
            )
            
            if analysis_type == "Local - Giải thích mẫu cụ thể":
                sample_for_analysis = st.number_input(
                    "Chọn mẫu để phân tích:",
                    0, 100, 0,
                    key="analysis_sample"
                )
            
            if st.button("🤖 Tạo Phân Tích AI", use_container_width=True, type="primary"):
                with st.spinner("AI đang phân tích SHAP values..."):
                    if analysis_type == "Global - Tổng quan mô hình":
                        ai_response = f"""
                        **🌍 Phân Tích Global - Tổng Quan Mô Hình {st.session_state.model_type}**
                        
                        **📊 Đặc trưng quan trọng nhất:**
                        
                        1. **{features[0]}** (Impact: {importance_values[0]:.3f})
                           - Đây là đặc trưng quan trọng nhất đối với mô hình
                           - Giá trị cao của đặc trưng này thường tăng xác suất vỡ nợ
                           - Chiếm {importance_values[0]/importance_values.sum()*100:.1f}% tổng impact
                        
                        2. **{features[1]}** (Impact: {importance_values[1]:.3f})
                           - Đặc trưng quan trọng thứ 2
                           - Có mối quan hệ phi tuyến với kết quả dự đoán
                        
                        3. **{features[2]}** (Impact: {importance_values[2]:.3f})
                           - Ảnh hưởng vừa phải nhưng ổn định
                        
                        **💡 Nhận xét:**
                        
                        - Top 3 đặc trưng chiếm {(importance_values[:3].sum()/importance_values.sum()*100):.1f}% tổng impact
                        - Mô hình phụ thuộc nhiều vào {features[0]}, cần đảm bảo chất lượng dữ liệu của biến này
                        - Các biến tài chính có xu hướng quan trọng hơn các biến nhân khẩu học
                        
                        **🎯 Khuyến nghị:**
                        
                        1. Tập trung thu thập và đảm bảo chất lượng của top features
                        2. Xem xét feature engineering cho các biến quan trọng
                        3. Giám sát sự thay đổi của feature importance theo thời gian
                        
                        ⚡ *Đây là phân tích mô phỏng. Backend sẽ tích hợp LLM để phân tích chi tiết.*
                        """
                    else:
                        ai_response = f"""
                        **🎯 Phân Tích Local - Mẫu #{sample_for_analysis}**
                        
                        **📋 Thông tin dự đoán:**
                        - Xác suất: {np.random.uniform(0.3, 0.9):.1%}
                        - Phân loại: {"✅ Tín dụng tốt" if np.random.random() > 0.5 else "⚠️ Rủi ro cao"}
                        
                        **🔍 Các yếu tố chính:**
                        
                        **Tác động tích cực (giảm rủi ro):**
                        • {features[0]}: Giá trị cao hơn trung bình, giúp giảm 15% xác suất vỡ nợ
                        • {features[1]}: Trong khoảng an toàn, đóng góp tích cực
                        
                        **Tác động tiêu cực (tăng rủi ro):**
                        • {features[2]}: Giá trị thấp bất thường, làm tăng 20% xác suất vỡ nợ
                        • {features[3]}: Vượt ngưỡng cảnh báo, cần xem xét kỹ
                        
                        **💭 Tổng kết:**
                        
                        Mẫu này có {"rủi ro thấp" if np.random.random() > 0.5 else "rủi ro cao"} do ảnh hưởng tổng hợp của các yếu tố.
                        Yếu tố quyết định chính là {features[0]}.
                        
                        **💡 Gợi ý cải thiện:**
                        1. Tăng giá trị của {features[2]} lên mức trung bình
                        2. Giảm {features[3]} xuống dưới ngưỡng cảnh báo
                        3. Duy trì {features[0]} ở mức hiện tại
                        
                        ⚡ *Đây là phân tích mô phỏng.*
                        """
                    
                    show_llm_analysis("Phân tích SHAP values", ai_response)
        
        with col2:
            st.markdown("#### ⚙️ Cấu Hình AI")
            
            st.markdown("""
            <div style="background-color: #262730; padding: 1rem; border-radius: 8px;">
                <h4 style="margin-top: 0; color: #667eea;">🤖 LLM Settings</h4>
                <p style="font-size: 0.9rem; margin-bottom: 0;">
                    Backend sẽ tích hợp LLM để:<br><br>
                    • Diễn giải SHAP values<br>
                    • Giải thích mối quan hệ giữa features<br>
                    • Đưa ra gợi ý cải thiện<br>
                    • Tạo báo cáo tự động<br>
                    • Trả lời câu hỏi về mô hình
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # LLM provider selection (placeholder)
            llm_provider = st.selectbox(
                "LLM Provider:",
                ["OpenAI GPT-4", "Anthropic Claude", "Local LLM"],
                key="llm_provider"
            )
            
            temperature = st.slider(
                "Temperature:",
                0.0, 1.0, 0.3, 0.1,
                key="llm_temp"
            )
            
            max_tokens = st.number_input(
                "Max tokens:",
                100, 2000, 500,
                key="llm_tokens"
            )
            
            st.markdown("---")
            show_processing_placeholder("Tích hợp LLM API cho phân tích tự động")
        
        st.markdown("---")
        
        # Interactive Q&A
        st.markdown("#### 💬 Hỏi Đáp Về Mô Hình")
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <p style="margin: 0;">💡 Đặt câu hỏi về mô hình và nhận câu trả lời từ AI dựa trên SHAP analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        user_question = st.text_input(
            "Câu hỏi của bạn:",
            placeholder="Ví dụ: Tại sao mô hình dự đoán mẫu này có rủi ro cao?",
            key="user_question"
        )
        
        if st.button("💬 Gửi Câu Hỏi", key="send_question"):
            if user_question:
                with st.spinner("🤖 AI đang suy nghĩ..."):
                    mock_answer = f"""
                    **Câu hỏi:** {user_question}
                    
                    **Trả lời:**
                    
                    Dựa trên phân tích SHAP, tôi có thể giải thích như sau:
                    
                    Mô hình {st.session_state.model_type} đưa ra dự đoán dựa trên sự kết hợp của nhiều yếu tố. 
                    Trong trường hợp này, yếu tố quan trọng nhất là {features[0]}, với SHAP value {importance_values[0]:.3f}.
                    
                    Các yếu tố khác cũng đóng góp vào quyết định cuối cùng theo thứ tự quan trọng giảm dần.
                    
                    💡 *Đây là câu trả lời mô phỏng. Backend sẽ tích hợp LLM để trả lời chính xác.*
                    """
                    
                    st.markdown(f"""
                    <div style="background-color: #1e3c72; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                        {mock_answer}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Vui lòng nhập câu hỏi!")

