"""
Trang Giải Thích SHAP - Model Explanation with SHAP
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
# UI components imported as needed
from utils.session_state import init_session_state

def render():
    """Render trang giải thích SHAP"""
    init_session_state()
    
    # Check view-only mode
    from utils.permissions import check_and_show_view_only
    is_view_only = check_and_show_view_only("💡 Model Explanation")
    
    st.markdown("## 🔍 Giải Thích Mô Hình Với SHAP")
    st.markdown("Hiểu rõ cách mô hình đưa ra quyết định thông qua SHAP (SHapley Additive exPlanations).")
    
    # Check prerequisites
    if st.session_state.data is None:
        st.warning("⚠️ Chưa có dữ liệu. Vui lòng upload dữ liệu trước.")
        return
    
    if st.session_state.model is None:
        st.warning("⚠️ Chưa có mô hình. Vui lòng huấn luyện mô hình trước.")
        return
    
    # Get the current model name
    current_model_name = st.session_state.get('selected_model_name', None)
    if current_model_name is None:
        current_model_name = st.session_state.get('model_type_select', 'Unknown')
    
    st.success(f"✅ Đang phân tích mô hình: {current_model_name}")
    
    st.markdown("---")
    
    # Get features and data
    features = st.session_state.selected_features
    target_col = st.session_state.target_column
    
    # Prepare data for SHAP
    if 'train_data' in st.session_state and st.session_state.train_data is not None:
        X_data = st.session_state.train_data[features]
    else:
        X_data = st.session_state.data[features]
    
    # SHAP explainer initialization
    if st.session_state.get('shap_explainer_obj') is None or st.session_state.get('shap_values_computed') is None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Sử dụng placeholder để tránh nút bị nhân đôi khi đang xử lý
            button_placeholder = st.empty()
            
            if button_placeholder.button("🔄 Khởi Tạo SHAP Explainer", key="init_shap_btn", type="primary", width='stretch', disabled=is_view_only):
                # Xóa nút và thay bằng spinner
                button_placeholder.empty()
                
                try:
                    with st.spinner("Đang tính toán SHAP values... (có thể mất vài phút)"):
                        from backend.explainability import initialize_shap_explainer
                        
                        # Initialize SHAP explainer
                        explainer, shap_values, X_explained = initialize_shap_explainer(
                            st.session_state.model,
                            X_data,
                            current_model_name
                        )
                        
                        # Save to session state
                        st.session_state.shap_explainer_obj = explainer
                        st.session_state.shap_values_computed = shap_values
                        st.session_state.shap_X_explained = X_explained
                        st.session_state.shap_feature_importance = explainer.get_feature_importance()
                        st.session_state.shap_expected_value = explainer.expected_value
                        
                        # Also update the placeholder values
                        st.session_state.explainer = "initialized"
                        st.session_state.shap_values = "computed"
                        
                        st.success("✅ Đã tính toán SHAP values thành công!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Lỗi khi tính SHAP: {str(e)}")
                    import traceback
                    with st.expander("Chi tiết lỗi"):
                        st.code(traceback.format_exc())
        
        st.info("💡 Nhấn nút trên để tính toán SHAP values cho mô hình")
        return
    
    # Get computed SHAP data
    shap_values = st.session_state.shap_values_computed
    X_explained = st.session_state.shap_X_explained
    feature_importance_df = st.session_state.shap_feature_importance
    expected_value = st.session_state.shap_expected_value
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🌍 Global Explanation",
        "🎯 Local Explanation",
        "✨ AI Interpretation"
    ])
    
    # Tab 1: Global Explanation
    with tab1:
        st.markdown("### 🌍 Global Feature Importance")
        st.markdown("Mức độ quan trọng tổng thể của các đặc trưng đối với mô hình.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📊 SHAP Summary Plot")
            
            # Create summary plot (bar chart) using real SHAP values
            fig = go.Figure()
            
            # Sort by importance
            plot_df = feature_importance_df.head(15).sort_values('Importance', ascending=True)
            
            fig.add_trace(go.Bar(
                y=plot_df['Feature'],
                x=plot_df['Importance'],
                orientation='h',
                marker=dict(
                    color=plot_df['Importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Impact")
                ),
                text=[f"{val:.3f}" for val in plot_df['Importance']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance - Global",
                xaxis_title="Mean |SHAP value|",
                yaxis_title="Features",
                template="plotly_dark",
                height=max(400, len(plot_df) * 35),
                showlegend=False
            )
            
            st.plotly_chart(fig, width='stretch')
            
            st.info("💡 **Giải thích**: Biểu đồ này cho thấy mức độ ảnh hưởng trung bình của mỗi đặc trưng đến dự đoán của mô hình.")
        
        with col2:
            st.markdown("#### 📋 Top Features")
            
            st.dataframe(
                feature_importance_df.style.format({'Importance': '{:.4f}'})
                .background_gradient(subset=['Importance'], cmap='Reds'),
                width='stretch',
                height=400
            )
            
            # Download
            csv = feature_importance_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Tải SHAP Values",
                csv,
                "shap_importance.csv",
                "text/csv"
            )
        
        st.markdown("---")
        
        # SHAP Beeswarm/Summary plot
        st.markdown("#### 🎻 SHAP Value Distribution")
        
        selected_feature = st.selectbox(
            "Chọn đặc trưng để xem phân phối:",
            features,
            key="global_feature_select"
        )
        
        # Get feature index
        feature_idx = features.index(selected_feature)
        
        # Get SHAP values and feature values for this feature
        shap_values_feature = shap_values[:, feature_idx]
        feature_values = X_explained[selected_feature].values
        
        fig = go.Figure()
        
        # Scatter plot with color based on feature value
        fig.add_trace(go.Scatter(
            x=shap_values_feature,
            y=np.random.randn(len(shap_values_feature)) * 0.1,  # Add jitter for visibility
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
            text=[f"Value: {v:.2f}<br>SHAP: {s:.3f}" for v, s in zip(feature_values, shap_values_feature)],
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
        
        st.plotly_chart(fig, width='stretch')
        
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
        
        max_samples = len(X_explained) - 1
        
        with col1:
            st.markdown("#### 📋 Chọn Mẫu")
            
            sample_selection_method = st.radio(
                "Phương pháp chọn:",
                ["Chọn theo index", "Chọn ngẫu nhiên"],
                key="sample_method"
            )
            
            if sample_selection_method == "Chọn theo index":
                sample_idx = st.number_input(
                    "Index mẫu:",
                    0, max_samples, 0,
                    key="sample_idx"
                )
            else:
                if st.button("🎲 Chọn Ngẫu Nhiên", key="random_sample"):
                    sample_idx = np.random.randint(0, max_samples + 1)
                    st.session_state.current_sample_idx = sample_idx
                sample_idx = st.session_state.get('current_sample_idx', 0)
            
            st.markdown(f"**Mẫu đang xem: #{sample_idx}**")
            
            # Get local SHAP values
            sample_shap = shap_values[sample_idx]
            sample_features = X_explained.iloc[sample_idx]
            
            # Calculate prediction
            base_value = float(expected_value) if isinstance(expected_value, (int, float, np.number)) else 0.5
            prediction = base_value + sample_shap.sum()
            pred_proba = 1 / (1 + np.exp(-prediction))  # Sigmoid for probability
            
            st.markdown("---")
            st.markdown("#### 🎯 Dự Đoán")
            
            st.metric("Xác suất", f"{pred_proba:.1%}")
            st.metric("Phân loại", "✅ Good" if pred_proba < 0.5 else "⚠️ Risk")
            st.metric("Base Value", f"{base_value:.3f}")
        
        with col2:
            st.markdown("#### 💧 SHAP Waterfall Plot")
            
            # Sort features by absolute SHAP value
            sorted_indices = np.argsort(np.abs(sample_shap))[::-1][:10]
            top_features = [features[i] for i in sorted_indices]
            top_shap = [sample_shap[i] for i in sorted_indices]
            top_values = [sample_features.iloc[i] for i in sorted_indices]
            
            # Create waterfall-like bar chart
            colors = ['#ff4444' if v < 0 else '#44bb44' for v in top_shap]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=top_shap,
                y=[f"{feat} = {val:.2f}" for feat, val in zip(top_features, top_values)],
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.3f}" for v in top_shap],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"SHAP Waterfall - Sample #{sample_idx}",
                xaxis_title="SHAP value (impact on prediction)",
                yaxis_title="Feature = Value",
                template="plotly_dark",
                height=450,
                showlegend=False
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Summary
            st.markdown(f"""
            <div style="background-color: #262730; padding: 1rem; border-radius: 8px;">
                <p style="margin: 0;">
                    <strong>📊 Tổng kết:</strong><br>
                    • Base value: {base_value:.3f}<br>
                    • Tổng SHAP: {sample_shap.sum():+.3f}<br>
                    • Prediction: {prediction:.3f} → Xác suất: {pred_proba:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature contributions table
        st.markdown("#### 📊 Chi Tiết Đóng Góp Của Các Đặc Trưng")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### ⬆️ Top Tác Động Tích Cực (Tăng rủi ro)")
            
            positive_mask = sample_shap > 0
            if positive_mask.any():
                pos_indices = np.where(positive_mask)[0]
                pos_sorted = pos_indices[np.argsort(sample_shap[pos_indices])[::-1]][:5]
                
                pos_data = [{
                    'Feature': features[i],
                    'Value': sample_features.iloc[i],
                    'SHAP Value': sample_shap[i]
                } for i in pos_sorted]
                
                pos_df = pd.DataFrame(pos_data)
                st.dataframe(
                    pos_df.style.format({
                        'SHAP Value': '{:+.4f}',
                        'Value': '{:.2f}'
                    }).background_gradient(subset=['SHAP Value'], cmap='Reds'),
                    width='stretch'
                )
            else:
                st.info("Không có tác động tích cực")
        
        with col2:
            st.markdown("##### ⬇️ Top Tác Động Tiêu Cực (Giảm rủi ro)")
            
            negative_mask = sample_shap < 0
            if negative_mask.any():
                neg_indices = np.where(negative_mask)[0]
                neg_sorted = neg_indices[np.argsort(sample_shap[neg_indices])][:5]
                
                neg_data = [{
                    'Feature': features[i],
                    'Value': sample_features.iloc[i],
                    'SHAP Value': sample_shap[i]
                } for i in neg_sorted]
                
                neg_df = pd.DataFrame(neg_data)
                st.dataframe(
                    neg_df.style.format({
                        'SHAP Value': '{:+.4f}',
                        'Value': '{:.2f}'
                    }).background_gradient(subset=['SHAP Value'], cmap='Greens'),
                    width='stretch'
                )
            else:
                st.info("Không có tác động tiêu cực")
        
        # All features table
        st.markdown("---")
        st.markdown("#### 📋 Tất Cả Đặc Trưng")
        
        all_contributions = pd.DataFrame({
            'Feature': features,
            'Value': [sample_features.iloc[i] for i in range(len(features))],
            'SHAP Value': sample_shap
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        st.dataframe(
            all_contributions.style.format({
                'SHAP Value': '{:+.4f}',
                'Value': '{:.2f}'
            }),
            width='stretch',
            height=300
        )
    
    # Tab 3: AI Interpretation
    with tab3:
        # Professional header with gradient
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
                    border: 1px solid rgba(102, 126, 234, 0.3);">
            <h3 style="margin: 0 0 0.5rem 0; color: #fff;">✨ Giải Thích Bằng AI</h3>
            <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 0.95rem;">
                Phân tích và diễn giải kết quả SHAP bằng ngôn ngữ tự nhiên với AI.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Import SHAP Analyzer
        from backend.llm_integration import create_shap_analyzer, LLMConfig
        
        # Check API configuration (Google Gemini) - only show warning if not configured
        api_configured = LLMConfig.GOOGLE_API_KEY is not None
        if not api_configured:
            st.warning("⚠️ Chưa cấu hình GOOGLE_API_KEY. Sử dụng phân tích tự động (hạn chế). Xem file `env.example` để cấu hình.")
        
        # Get model name
        model_name_display = st.session_state.get('selected_model_name', st.session_state.get('model_type_select', 'Unknown'))
        
        # Convert expected_value to scalar for display
        exp_val_display = float(expected_value[1]) if isinstance(expected_value, np.ndarray) and len(expected_value) > 1 else (float(expected_value[0]) if isinstance(expected_value, np.ndarray) else float(expected_value))
        
        # SHAP Statistics Cards - 3 columns with gradient cards
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                        padding: 1.2rem; border-radius: 10px; text-align: center;
                        border: 1px solid rgba(102, 126, 234, 0.3);">
                <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 0.85rem;">Số Mẫu Đã Tính</p>
                <h2 style="margin: 0.3rem 0 0 0; color: #fff; font-size: 1.8rem;">{len(X_explained)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #134e5e 0%, #71b280 100%); 
                        padding: 1.2rem; border-radius: 10px; text-align: center;
                        border: 1px solid rgba(113, 178, 128, 0.3);">
                <p style="margin: 0; color: rgba(255,255,255,0.7); font-size: 0.85rem;">Số Features</p>
                <h2 style="margin: 0.3rem 0 0 0; color: #fff; font-size: 1.8rem;">{len(features)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); 
                        padding: 1.2rem; border-radius: 10px; text-align: center;
                        border: 1px solid rgba(254, 202, 87, 0.3);">
                <p style="margin: 0; color: rgba(255,255,255,0.9); font-size: 0.85rem;">Mean |SHAP|</p>
                <h2 style="margin: 0.3rem 0 0 0; color: #fff; font-size: 1.8rem;">{np.abs(shap_values).mean():.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="background-color: rgba(38, 39, 48, 0.8); padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #667eea;">💬 Phân Tích Tự Động</h4>
            </div>
            """, unsafe_allow_html=True)
            
            analysis_type = st.radio(
                "Loại phân tích:",
                ["Global - Tổng quan mô hình", "Local - Giải thích mẫu cụ thể"],
                key="analysis_type",
                horizontal=True
            )
            
            if analysis_type == "Local - Giải thích mẫu cụ thể":
                sample_for_analysis = st.number_input(
                    "Chọn mẫu để phân tích:",
                    0, max_samples, 0,
                    key="analysis_sample"
                )
            
            if st.button("✨ Tạo Phân Tích AI", width='stretch', type="primary"):
                with st.spinner("✨ AI đang phân tích SHAP values..."):
                    try:
                        # Create SHAP Analyzer
                        shap_analyzer = create_shap_analyzer()
                        
                        if analysis_type == "Global - Tổng quan mô hình":
                            # Call AI for global analysis
                            ai_response = shap_analyzer.analyze_global(
                                model_name=model_name_display,
                                feature_importance=feature_importance_df,
                                shap_values=shap_values,
                                expected_value=exp_val_display,
                                features=features
                            )
                        else:
                            # Call AI for local analysis
                            ai_response = shap_analyzer.analyze_local(
                                model_name=model_name_display,
                                feature_importance=feature_importance_df,
                                shap_values=shap_values,
                                expected_value=exp_val_display,
                                features=features,
                                sample_data=X_explained,
                                sample_idx=sample_for_analysis
                            )
                        
                        # Store in session state
                        st.session_state.last_ai_analysis = ai_response
                        st.session_state.last_analysis_type = analysis_type
                        st.rerun()  # Refresh to show results properly
                        
                    except Exception as e:
                        ai_response = f"❌ Lỗi khi phân tích: {str(e)}"
                        st.session_state.last_ai_analysis = ai_response
                        st.rerun()
            
            # Display last analysis
            if 'last_ai_analysis' in st.session_state and st.session_state.last_ai_analysis:
                with st.expander("📋 Kết Quả Phân Tích AI", expanded=True):
                    st.markdown(st.session_state.last_ai_analysis)
        
        with col2:
            st.markdown("""
            <div style="background-color: rgba(38, 39, 48, 0.8); padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #667eea;">📊 Chi Tiết SHAP</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: linear-gradient(180deg, #1f2937 0%, #262730 100%); 
                        padding: 1rem; border-radius: 8px; border: 1px solid rgba(102, 126, 234, 0.2);">
                <table style="width: 100%; color: rgba(255,255,255,0.85); font-size: 0.9rem;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <td style="padding: 0.5rem 0;">Expected Value</td>
                        <td style="padding: 0.5rem 0; text-align: right; font-weight: 600;">{exp_val_display:.4f}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <td style="padding: 0.5rem 0;">Max |SHAP|</td>
                        <td style="padding: 0.5rem 0; text-align: right; font-weight: 600;">{np.abs(shap_values).max():.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem 0;">Model</td>
                        <td style="padding: 0.5rem 0; text-align: right; font-weight: 600; color: #667eea;">{model_name_display}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Recalculate button
            if st.button("🔄 Tính Lại SHAP", width='stretch', disabled=is_view_only):
                st.session_state.shap_explainer_obj = None
                st.session_state.shap_values_computed = None
                st.session_state.explainer = None
                st.session_state.shap_values = None
                st.session_state.last_ai_analysis = None
                st.session_state.shap_chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        # Interactive Chat Q&A with improved styling
        st.markdown("""
        <div style="background-color: rgba(38, 39, 48, 0.8); padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="margin: 0 0 0.3rem 0; color: #667eea;">💬 Chat Với AI Về Mô Hình</h4>
            <p style="margin: 0; color: rgba(255,255,255,0.6); font-size: 0.85rem;">
                Đặt câu hỏi về mô hình và nhận câu trả lời từ AI dựa trên SHAP analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize chat history
        if 'shap_chat_history' not in st.session_state:
            st.session_state.shap_chat_history = []
        
        # Display chat history with improved styling
        if st.session_state.shap_chat_history:
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.shap_chat_history:
                    if msg['role'] == 'user':
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                                    padding: 1rem; border-radius: 10px; margin-bottom: 0.8rem;
                                    border-left: 4px solid #667eea;">
                            <strong style="color: #a8c0ff;">🧑 Bạn:</strong>
                            <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">{msg['content']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1f2937 0%, #2d3748 100%); 
                                    padding: 1rem; border-radius: 10px; margin-bottom: 0.8rem;
                                    border-left: 4px solid #48bb78;">
                            <strong style="color: #9ae6b4;">✨ AI:</strong>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(msg['content'])
        
        # Chat input with better styling
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_question = st.text_input(
                "Câu hỏi của bạn:",
                placeholder="Ví dụ: Tại sao loan_term_months là feature quan trọng nhất?",
                key="user_question",
                label_visibility="collapsed"
            )
        with col_btn:
            send_clicked = st.button("📤 Gửi", key="send_question", width='stretch', type="primary")
        
        if send_clicked and user_question:
            with st.spinner("✨ AI đang suy nghĩ..."):
                try:
                    # Create SHAP Analyzer
                    shap_analyzer = create_shap_analyzer()
                    
                    # Call AI chat
                    ai_answer = shap_analyzer.chat(
                        user_question=user_question,
                        model_name=model_name_display,
                        feature_importance=feature_importance_df,
                        shap_values=shap_values,
                        expected_value=exp_val_display,
                        features=features,
                        sample_data=X_explained,
                        conversation_history=st.session_state.shap_chat_history
                    )
                    
                    # Add to history
                    st.session_state.shap_chat_history.append({
                        'role': 'user',
                        'content': user_question
                    })
                    st.session_state.shap_chat_history.append({
                        'role': 'assistant',
                        'content': ai_answer
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
        elif send_clicked and not user_question:
            st.warning("Vui lòng nhập câu hỏi!")
        
        # Clear chat button
        if st.session_state.shap_chat_history:
            if st.button("🗑️ Xóa Lịch Sử Chat", key="clear_chat"):
                st.session_state.shap_chat_history = []
                st.rerun()
        
        # Sample questions with better card styling
        st.markdown("---")
        st.markdown("##### 💡 Câu Hỏi Gợi Ý")
        
        sample_questions = [
            "Tại sao feature quan trọng nhất lại ảnh hưởng nhiều đến dự đoán?",
            "Mô hình có thể có bias không? Giải thích.",
            "Làm sao để cải thiện độ chính xác dựa trên SHAP analysis?",
            "So sánh tác động của top 3 features",
            "Khách hàng cần làm gì để giảm rủi ro tín dụng?"
        ]
        
        cols = st.columns(2)
        for i, q in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(f"💬 {q[:40]}...", key=f"sample_q_{i}", width='stretch'):
                    st.session_state.sample_question_selected = q
                    st.rerun()
        
        # Handle sample question selection
        if 'sample_question_selected' in st.session_state and st.session_state.sample_question_selected:
            selected_q = st.session_state.sample_question_selected
            st.session_state.sample_question_selected = None  # Clear
            
            with st.spinner("✨ AI đang suy nghĩ..."):
                try:
                    shap_analyzer = create_shap_analyzer()
                    
                    ai_answer = shap_analyzer.chat(
                        user_question=selected_q,
                        model_name=model_name_display,
                        feature_importance=feature_importance_df,
                        shap_values=shap_values,
                        expected_value=exp_val_display,
                        features=features,
                        sample_data=X_explained,
                        conversation_history=st.session_state.shap_chat_history
                    )
                    
                    st.session_state.shap_chat_history.append({
                        'role': 'user',
                        'content': selected_q
                    })
                    st.session_state.shap_chat_history.append({
                        'role': 'assistant',
                        'content': ai_answer
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")

