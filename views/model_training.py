"""
Trang Huấn Luyện Mô Hình - Model Training
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from utils.ui_components import show_processing_placeholder
from utils.session_state import init_session_state

def render():
    """Render trang huấn luyện mô hình"""
    init_session_state()
    
    st.markdown("## 🤖 Huấn Luyện Mô Hình")
    st.markdown("Chọn và cấu hình mô hình Machine Learning để dự đoán điểm tín dụng.")
    
    # Check prerequisites
    if st.session_state.data is None:
        st.warning("⚠️ Chưa có dữ liệu. Vui lòng upload dữ liệu từ trang 'Upload & EDA'.")
        return
    
    if not st.session_state.selected_features:
        st.warning("⚠️ Chưa chọn đặc trưng. Vui lòng chọn đặc trưng từ trang 'Xử Lý & Chọn Biến'.")
        return
    
    st.success(f"✅ Sẵn sàng huấn luyện với {len(st.session_state.selected_features)} đặc trưng")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "⚙️ Cấu Hình Mô Hình",
        "📊 Kết Quả Đánh Giá",
        "📈 So Sánh Mô Hình"
    ])
    
    # Tab 1: Model Configuration
    with tab1:
        st.markdown("### ⚙️ Cấu Hình Và Huấn Luyện")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 1️⃣ Chọn Mô Hình")
            
            model_type = st.selectbox(
                "Loại mô hình:",
                [
                    "Logistic Regression",
                    "Random Forest",
                    "XGBoost",
                    "LightGBM",
                    "CatBoost",
                    "Gradient Boosting"
                ],
                key="model_type"
            )
            
            st.markdown("---")
            
            st.markdown("#### 2️⃣ Chia Dữ Liệu")
            
            test_size = st.slider(
                "Tỷ lệ test set (%):",
                10, 40, 20, 5,
                key="test_size"
            )
            
            random_state = st.number_input(
                "Random state:",
                0, 1000, 42,
                key="random_state"
            )
            
            stratify = st.checkbox(
                "Stratified split",
                value=True,
                help="Giữ nguyên tỷ lệ các class trong train/test"
            )
            
            st.markdown("---")
            
            st.markdown("#### 3️⃣ Tham Số Mô Hình")
            
            # Model-specific parameters
            if model_type == "Logistic Regression":
                c_value = st.slider("C (Regularization):", 0.001, 10.0, 1.0, 0.001, key="lr_c")
                max_iter = st.number_input("Max iterations:", 100, 1000, 200, key="lr_iter")
                
            elif model_type in ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"]:
                n_estimators = st.slider("Số cây (n_estimators):", 50, 500, 100, 10, key="n_trees")
                max_depth = st.slider("Độ sâu tối đa:", 3, 20, 6, 1, key="max_depth")
                learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01, key="lr")
                
                if model_type in ["XGBoost", "LightGBM", "CatBoost"]:
                    subsample = st.slider("Subsample:", 0.5, 1.0, 0.8, 0.1, key="subsample")
            
            st.markdown("---")
            
            # Train button
            if st.button("🚀 Huấn Luyện Mô Hình", type="primary", use_container_width=True):
                with st.spinner(f"Đang huấn luyện {model_type}..."):
                    show_processing_placeholder(f"Train {model_type} với {len(st.session_state.selected_features)} features")
                    
                    # Save model info to session
                    st.session_state.model_type = model_type
                    st.session_state.model = "trained"  # Placeholder
                    
                    # Mock metrics
                    st.session_state.model_metrics = {
                        'accuracy': np.random.uniform(0.75, 0.95),
                        'precision': np.random.uniform(0.70, 0.90),
                        'recall': np.random.uniform(0.65, 0.88),
                        'f1': np.random.uniform(0.68, 0.89),
                        'auc': np.random.uniform(0.80, 0.96)
                    }
                    
                    st.success(f"✅ Đã huấn luyện {model_type} thành công!")
                    st.balloons()
        
        with col2:
            st.markdown("#### 📋 Thông Tin Huấn Luyện")
            
            # Training info panel
            st.markdown("""
            <div style="background-color: #262730; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="margin-top: 0; color: #667eea;">💡 Hướng Dẫn</h4>
                <ul style="margin-bottom: 0;">
                    <li><strong>Logistic Regression</strong>: Nhanh, dễ giải thích, phù hợp với dữ liệu tuyến tính</li>
                    <li><strong>Random Forest</strong>: Mạnh mẽ, chống overfitting tốt</li>
                    <li><strong>XGBoost/LightGBM</strong>: Hiệu suất cao, thường cho kết quả tốt nhất</li>
                    <li><strong>CatBoost</strong>: Tốt với biến phân loại, ít cần tiền xử lý</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Current configuration summary
            if st.session_state.model is not None:
                st.markdown("#### ✅ Mô Hình Hiện Tại")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 10px;">
                    <h3 style="margin: 0; color: white;">{st.session_state.model_type}</h3>
                    <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">
                        Đã huấn luyện với {len(st.session_state.selected_features)} features
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Quick metrics
                metrics = st.session_state.model_metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                with col3:
                    st.metric("AUC", f"{metrics['auc']:.3f}")
            else:
                st.info("⏳ Chưa huấn luyện mô hình nào")
            
            # Additional options
            st.markdown("#### ⚙️ Tùy Chọn Nâng Cao")
            
            with st.expander("Cross-Validation"):
                cv_folds = st.slider("Số folds:", 3, 10, 5, key="cv_folds")
                if st.button("🔄 Chạy Cross-Validation", key="run_cv"):
                    show_processing_placeholder(f"Cross-validation với {cv_folds} folds")
                    st.success(f"✅ CV Score: {np.random.uniform(0.75, 0.90):.3f} (+/- {np.random.uniform(0.02, 0.05):.3f})")
            
            with st.expander("Hyperparameter Tuning"):
                tuning_method = st.selectbox(
                    "Phương pháp:",
                    ["Grid Search", "Random Search", "Bayesian Optimization"],
                    key="tuning_method"
                )
                if st.button("🔍 Tìm Tham Số Tốt Nhất", key="tune_params"):
                    show_processing_placeholder(f"Hyperparameter tuning với {tuning_method}")
                    st.success("✅ Đã tìm được tham số tốt nhất!")
    
    # Tab 2: Evaluation Results
    with tab2:
        st.markdown("### 📊 Kết Quả Đánh Giá Mô Hình")
        
        if st.session_state.model is None:
            st.warning("⚠️ Chưa có mô hình được huấn luyện. Vui lòng huấn luyện mô hình trước.")
            return
        
        metrics = st.session_state.model_metrics
        
        # Metrics overview
        st.markdown("#### 📈 Các Chỉ Số Đánh Giá")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.3f}")
        with col5:
            st.metric("AUC", f"{metrics['auc']:.3f}")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Confusion Matrix")
            
            # Mock confusion matrix
            cm = np.array([
                [850, 150],
                [100, 900]
            ])
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Negative (0)', 'Positive (1)'],
                y=['Negative (0)', 'Positive (1)'],
                text_auto=True,
                color_continuous_scale='Blues',
                title="Confusion Matrix"
            )
            
            fig.update_layout(
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics explanation
            st.markdown("""
            <div style="background-color: #262730; padding: 1rem; border-radius: 8px;">
                <p style="margin: 0; font-size: 0.9rem;">
                    <strong>True Negative:</strong> {0} | <strong>False Positive:</strong> {1}<br>
                    <strong>False Negative:</strong> {2} | <strong>True Positive:</strong> {3}
                </p>
            </div>
            """.format(cm[0,0], cm[0,1], cm[1,0], cm[1,1]), unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📈 ROC Curve")
            
            # Mock ROC curve
            fpr = np.linspace(0, 1, 100)
            tpr = np.sqrt(fpr) * metrics['auc'] + np.random.normal(0, 0.02, 100)
            tpr = np.clip(tpr, 0, 1)
            
            fig = go.Figure()
            
            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{st.session_state.model_type} (AUC = {metrics["auc"]:.3f})',
                line=dict(color='#667eea', width=3)
            ))
            
            # Diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random (AUC = 0.5)',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                template="plotly_dark",
                height=400,
                showlegend=True,
                legend=dict(x=0.6, y=0.1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed metrics table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Classification Report")
            
            report_data = pd.DataFrame({
                'Class': ['Negative (0)', 'Positive (1)', 'Weighted Avg'],
                'Precision': [
                    metrics['precision'] - 0.02,
                    metrics['precision'] + 0.02,
                    metrics['precision']
                ],
                'Recall': [
                    metrics['recall'] + 0.01,
                    metrics['recall'] - 0.01,
                    metrics['recall']
                ],
                'F1-Score': [
                    metrics['f1'] - 0.01,
                    metrics['f1'] + 0.01,
                    metrics['f1']
                ],
                'Support': [1000, 1000, 2000]
            })
            
            st.dataframe(
                report_data.style.format({
                    'Precision': '{:.3f}',
                    'Recall': '{:.3f}',
                    'F1-Score': '{:.3f}'
                }).background_gradient(subset=['Precision', 'Recall', 'F1-Score'], cmap='Greens'),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 📊 Precision-Recall Curve")
            
            # Mock PR curve
            recall_vals = np.linspace(0, 1, 100)
            precision_vals = 1 - recall_vals * 0.3 + np.random.normal(0, 0.02, 100)
            precision_vals = np.clip(precision_vals, 0, 1)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=recall_vals,
                y=precision_vals,
                mode='lines',
                fill='tozeroy',
                name=st.session_state.model_type,
                line=dict(color='#764ba2', width=3)
            ))
            
            fig.update_layout(
                title='Precision-Recall Curve',
                xaxis_title='Recall',
                yaxis_title='Precision',
                template="plotly_dark",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("💾 Lưu Mô Hình", use_container_width=True):
                show_processing_placeholder("Lưu mô hình vào file .pkl")
                st.success("✅ Đã lưu mô hình!")
    
    # Tab 3: Model Comparison
    with tab3:
        st.markdown("### 📈 So Sánh Nhiều Mô Hình")
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p style="margin: 0;">💡 Huấn luyện và so sánh nhiều mô hình để chọn ra mô hình tốt nhất.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection for comparison
        models_to_compare = st.multiselect(
            "Chọn các mô hình để so sánh:",
            [
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "LightGBM",
                "CatBoost"
            ],
            default=["Logistic Regression", "Random Forest", "XGBoost"]
        )
        
        if st.button("🔄 So Sánh Các Mô Hình", type="primary"):
            with st.spinner("Đang huấn luyện và so sánh..."):
                show_processing_placeholder(f"Train và so sánh {len(models_to_compare)} mô hình")
                st.success("✅ Đã hoàn thành so sánh!")
        
        if models_to_compare:
            st.markdown("---")
            st.markdown("#### 📊 Kết Quả So Sánh")
            
            # Mock comparison data
            comparison_data = []
            for model in models_to_compare:
                comparison_data.append({
                    'Model': model,
                    'Accuracy': np.random.uniform(0.75, 0.92),
                    'Precision': np.random.uniform(0.70, 0.90),
                    'Recall': np.random.uniform(0.68, 0.88),
                    'F1-Score': np.random.uniform(0.70, 0.89),
                    'AUC': np.random.uniform(0.80, 0.95),
                    'Training Time (s)': np.random.uniform(1, 30)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display table
            st.dataframe(
                comparison_df.style.format({
                    'Accuracy': '{:.3f}',
                    'Precision': '{:.3f}',
                    'Recall': '{:.3f}',
                    'F1-Score': '{:.3f}',
                    'AUC': '{:.3f}',
                    'Training Time (s)': '{:.2f}'
                }).background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart comparison
                fig = go.Figure()
                
                metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
                
                for metric in metrics_to_plot:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=comparison_df['Model'],
                        y=comparison_df[metric],
                        text=comparison_df[metric].round(3),
                        textposition='outside'
                    ))
                
                fig.update_layout(
                    title='So Sánh Metrics Giữa Các Mô Hình',
                    xaxis_title='Model',
                    yaxis_title='Score',
                    template="plotly_dark",
                    height=400,
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # ROC curves comparison
                fig = go.Figure()
                
                colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
                
                for i, model in enumerate(models_to_compare):
                    auc = comparison_df[comparison_df['Model'] == model]['AUC'].values[0]
                    fpr = np.linspace(0, 1, 100)
                    tpr = np.sqrt(fpr) * auc + np.random.normal(0, 0.02, 100)
                    tpr = np.clip(tpr, 0, 1)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode='lines',
                        name=f'{model} (AUC={auc:.3f})',
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
                
                # Diagonal
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='So Sánh ROC Curves',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    template="plotly_dark",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Best model recommendation
            best_model_idx = comparison_df['AUC'].idxmax()
            best_model = comparison_df.loc[best_model_idx, 'Model']
            best_auc = comparison_df.loc[best_model_idx, 'AUC']
            
            st.success(f"🏆 **Mô hình tốt nhất**: {best_model} với AUC = {best_auc:.3f}")

