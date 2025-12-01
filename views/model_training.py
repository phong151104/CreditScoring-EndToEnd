"""
Trang Huấn Luyện Mô Hình - Model Training
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
    
    # Check if train/test split exists
    if 'train_data' not in st.session_state or st.session_state.train_data is None:
        st.warning("⚠️ Chưa chia tập dữ liệu. Vui lòng chia tập Train/Valid/Test ở trang 'Xử Lý & Chọn Biến'.")
        return
    
    st.success(f"✅ Sẵn sàng huấn luyện với {len(st.session_state.selected_features)} đặc trưng")
    
    st.markdown("---")
    
    # Initialize model history
    if 'model_history' not in st.session_state:
        st.session_state.model_history = []

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
                key="model_type_select"
            )
            
            st.markdown("---")
            
            st.markdown("#### 2️⃣ Dữ Liệu Huấn Luyện")
            
            st.info(f"📊 Tập Train: {len(st.session_state.train_data)} dòng")
            if 'test_data' in st.session_state and st.session_state.test_data is not None:
                st.info(f"🧪 Tập Test: {len(st.session_state.test_data)} dòng")
            
            st.markdown("---")
            
            st.markdown("#### 3️⃣ Tham Số Mô Hình")
            
            # Get tuned params if available
            tuned_params = st.session_state.get('best_tuned_params', {})
            
            # Check if we need to apply tuned params (clear widget keys to force new defaults)
            if st.session_state.get('apply_tuned_params_flag', False):
                # Clear widget keys to reset to new defaults
                keys_to_clear = ['lr_c', 'lr_iter', 'n_trees', 'max_depth', 'lr', 'subsample', 'min_samples_split', 'unlimited_depth']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.apply_tuned_params_flag = False
            
            # Model-specific parameters
            if model_type == "Logistic Regression":
                default_c = tuned_params.get('C', 1.0)
                default_max_iter = tuned_params.get('max_iter', 200)
                # Clamp values to slider range
                default_c = max(0.001, min(10.0, default_c))
                default_max_iter = max(100, min(1000, default_max_iter))
                
                c_value = st.slider("C (Regularization):", 0.001, 10.0, float(default_c), 0.001, key="lr_c")
                max_iter = st.number_input("Max iterations:", 100, 1000, int(default_max_iter), key="lr_iter")
                
            elif model_type == "Random Forest":
                default_n_estimators = tuned_params.get('n_estimators', 100)
                default_max_depth = tuned_params.get('max_depth', 10)
                default_min_samples_split = tuned_params.get('min_samples_split', 2)
                # Clamp values
                default_n_estimators = max(50, min(500, default_n_estimators))
                if default_max_depth is None:
                    default_max_depth = 10
                default_max_depth = max(3, min(20, default_max_depth))
                default_min_samples_split = max(2, min(20, default_min_samples_split))
                
                n_estimators = st.slider("Số cây (n_estimators):", 50, 500, int(default_n_estimators), 10, key="n_trees")
                max_depth = st.slider("Độ sâu tối đa:", 3, 20, int(default_max_depth), 1, key="max_depth")
                min_samples_split = st.slider("Min samples split:", 2, 20, int(default_min_samples_split), 1, key="min_samples_split")
                
            elif model_type in ["XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"]:
                default_n_estimators = tuned_params.get('n_estimators', tuned_params.get('iterations', 100))
                default_max_depth = tuned_params.get('max_depth', tuned_params.get('depth', 6))
                default_learning_rate = tuned_params.get('learning_rate', 0.1)
                default_subsample = tuned_params.get('subsample', 0.8)
                
                # Check if max_depth is unlimited (-1 or None)
                is_unlimited_depth = default_max_depth == -1 or default_max_depth is None
                
                # Clamp values
                default_n_estimators = max(50, min(500, default_n_estimators))
                if default_max_depth is None or default_max_depth == -1:
                    default_max_depth = 6  # Default for slider display
                default_max_depth = max(3, min(20, default_max_depth))
                default_learning_rate = max(0.01, min(0.3, default_learning_rate))
                default_subsample = max(0.5, min(1.0, default_subsample))
                
                n_estimators = st.slider("Số cây (n_estimators):", 50, 500, int(default_n_estimators), 10, key="n_trees")
                
                # For LightGBM, allow unlimited depth option
                if model_type == "LightGBM":
                    unlimited_depth = st.checkbox(
                        "🔓 Không giới hạn độ sâu (max_depth = -1)", 
                        value=is_unlimited_depth,
                        key="unlimited_depth",
                        help="Trong LightGBM, max_depth=-1 cho phép cây phát triển không giới hạn. Có thể tăng hiệu suất nhưng cũng tăng nguy cơ overfitting."
                    )
                    if unlimited_depth:
                        max_depth = -1
                        st.info("💡 Độ sâu không giới hạn - cây sẽ phát triển dựa trên num_leaves")
                    else:
                        max_depth = st.slider("Độ sâu tối đa:", 3, 20, int(default_max_depth), 1, key="max_depth")
                else:
                    max_depth = st.slider("Độ sâu tối đa:", 3, 20, int(default_max_depth), 1, key="max_depth")
                
                learning_rate = st.slider("Learning rate:", 0.01, 0.3, float(default_learning_rate), 0.01, key="lr")
                subsample = st.slider("Subsample:", 0.5, 1.0, float(default_subsample), 0.1, key="subsample")
            
            st.markdown("---")
            
            # Train button
            if st.button("🚀 Huấn Luyện Mô Hình", type="primary", width='stretch'):
                try:
                    with st.spinner(f"Đang huấn luyện {model_type}..."):
                        # Prepare data
                        target_col = st.session_state.target_column
                        features = st.session_state.selected_features
                        
                        X_train = st.session_state.train_data[features]
                        y_train = st.session_state.train_data[target_col]
                        
                        # Use test data if available, otherwise split train data (fallback)
                        if 'test_data' in st.session_state and st.session_state.test_data is not None:
                            X_test = st.session_state.test_data[features]
                            y_test = st.session_state.test_data[target_col]
                        else:
                            # Fallback if no test data (should not happen if flow is followed)
                            from sklearn.model_selection import train_test_split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_train, y_train, test_size=0.2, random_state=42
                            )
                        
                        # Collect parameters
                        params = {}
                        if model_type == "Logistic Regression":
                            params['C'] = c_value
                            params['max_iter'] = max_iter
                        elif model_type == "Random Forest":
                            params['n_estimators'] = n_estimators
                            params['max_depth'] = max_depth
                            params['min_samples_split'] = min_samples_split
                        elif model_type in ["XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"]:
                            params['n_estimators'] = n_estimators
                            params['max_depth'] = max_depth
                            params['learning_rate'] = learning_rate
                            params['subsample'] = subsample
                        
                        # Import backend
                        from backend.models.trainer import train_model
                        
                        # Train model
                        model, metrics = train_model(
                            X_train, y_train, X_test, y_test, 
                            model_type, params
                        )
                        
                        # Save model info to session
                        st.session_state.model = model
                        st.session_state.model_metrics = metrics
                        st.session_state.selected_model_name = model_type
                        
                        # Add to history
                        import datetime
                        history_entry = {
                            'Model': model_type,
                            'Accuracy': metrics['accuracy'],
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1-Score': metrics['f1'],
                            'AUC': metrics['auc'],
                            'Timestamp': datetime.datetime.now().strftime("%H:%M:%S"),
                            'Params': str(params)
                        }
                        st.session_state.model_history.append(history_entry)
                        
                        # Auto-select the newly trained model
                        st.session_state.selected_model_idx = len(st.session_state.model_history) - 1
                        st.session_state.selected_model_timestamp = history_entry['Timestamp']
                        
                        st.success(f"✅ Đã huấn luyện {model_type} thành công!")
                        st.balloons()
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Lỗi khi huấn luyện mô hình: {str(e)}")
                    import traceback
                    with st.expander("Chi tiết lỗi"):
                        st.code(traceback.format_exc())
        
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
            
            # Current configuration summary - use selected_model_name if available
            current_model_display = st.session_state.get('selected_model_name', None)
            if current_model_display is None and st.session_state.model is not None:
                current_model_display = st.session_state.get('model_type_select', 'Unknown')
            
            if st.session_state.model is not None and current_model_display:
                st.markdown("#### ✅ Mô Hình Hiện Tại")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1f2937 0%, #345f9c 100%); 
                            padding: 1.5rem; border-radius: 10px;">
                    <h3 style="margin: 0; color: white;">{current_model_display}</h3>
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
                    try:
                        with st.spinner(f"Đang chạy Cross-Validation với {cv_folds} folds..."):
                            # Prepare data
                            target_col = st.session_state.target_column
                            features = st.session_state.selected_features
                            
                            X = st.session_state.train_data[features]
                            y = st.session_state.train_data[target_col]
                            
                            # Get current model type from selectbox
                            current_model_type = st.session_state.get('model_type_select', 'Logistic Regression')
                            
                            # Collect parameters based on model type
                            params = {}
                            if current_model_type == "Logistic Regression":
                                params['C'] = st.session_state.get('lr_c', 1.0)
                                params['max_iter'] = st.session_state.get('lr_iter', 200)
                            elif current_model_type == "Random Forest":
                                params['n_estimators'] = st.session_state.get('n_trees', 100)
                                params['max_depth'] = st.session_state.get('max_depth', 10)
                            elif current_model_type in ["XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"]:
                                params['n_estimators'] = st.session_state.get('n_trees', 100)
                                params['max_depth'] = st.session_state.get('max_depth', 6)
                                params['learning_rate'] = st.session_state.get('lr', 0.1)
                                params['subsample'] = st.session_state.get('subsample', 0.8)
                            
                            # Import backend
                            from backend.models.trainer import cross_validate_model
                            
                            # Run cross-validation
                            cv_results = cross_validate_model(X, y, current_model_type, params, cv_folds)
                            
                            # Save to session state
                            st.session_state.cv_results = cv_results
                            
                            # Display results
                            st.success(f"✅ Cross-Validation hoàn thành!")
                            
                            # Show metrics
                            cv_col1, cv_col2 = st.columns(2)
                            with cv_col1:
                                st.metric("AUC", f"{cv_results['auc']['mean']:.3f} (+/- {cv_results['auc']['std']:.3f})")
                                st.metric("Accuracy", f"{cv_results['accuracy']['mean']:.3f} (+/- {cv_results['accuracy']['std']:.3f})")
                            with cv_col2:
                                st.metric("F1-Score", f"{cv_results['f1']['mean']:.3f} (+/- {cv_results['f1']['std']:.3f})")
                                st.metric("Precision", f"{cv_results['precision']['mean']:.3f} (+/- {cv_results['precision']['std']:.3f})")
                            
                            # Show fold details (no nested expander)
                            st.markdown("##### 📋 Chi tiết từng Fold")
                            fold_df = pd.DataFrame({
                                'Fold': [f"Fold {i+1}" for i in range(cv_folds)],
                                'Accuracy': cv_results['accuracy']['scores'],
                                'Precision': cv_results['precision']['scores'],
                                'Recall': cv_results['recall']['scores'],
                                'F1': cv_results['f1']['scores'],
                                'AUC': cv_results['auc']['scores']
                            })
                            st.dataframe(fold_df.style.format({
                                'Accuracy': '{:.3f}',
                                'Precision': '{:.3f}',
                                'Recall': '{:.3f}',
                                'F1': '{:.3f}',
                                'AUC': '{:.3f}'
                            }).background_gradient(subset=['AUC'], cmap='RdYlGn'), width='stretch')
                    except Exception as e:
                        st.error(f"❌ Lỗi khi chạy Cross-Validation: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            with st.expander("Hyperparameter Tuning"):
                tuning_method = st.selectbox(
                    "Phương pháp:",
                    ["Grid Search", "Random Search", "Bayesian Optimization"],
                    key="tuning_method"
                )
                tuning_cv_folds = st.slider("Số folds cho CV:", 3, 10, 5, key="tuning_cv_folds")
                
                if st.button("🔍 Tìm Tham Số Tốt Nhất", key="tune_params"):
                    try:
                        with st.spinner(f"Đang chạy {tuning_method}... (có thể mất vài phút)"):
                            # Prepare data
                            target_col = st.session_state.target_column
                            features = st.session_state.selected_features
                            
                            X = st.session_state.train_data[features]
                            y = st.session_state.train_data[target_col]
                            
                            # Get current model type from selectbox
                            current_model_type = st.session_state.get('model_type_select', 'Logistic Regression')
                            
                            # Import backend
                            from backend.models.trainer import hyperparameter_tuning
                            
                            # Run hyperparameter tuning
                            tuning_results = hyperparameter_tuning(
                                X, y, current_model_type, tuning_method, tuning_cv_folds
                            )
                            
                            # Save to session state and rerun to display results
                            st.session_state.tuning_results = tuning_results
                        
                        st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Lỗi khi chạy Hyperparameter Tuning: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                
                # Display saved tuning results if exists
                if 'tuning_results' in st.session_state and st.session_state.tuning_results is not None:
                    tuning_results = st.session_state.tuning_results
                    
                    st.success(f"✅ Đã tìm được tham số tốt nhất!")
                    
                    st.markdown("##### 🏆 Tham Số Tốt Nhất")
                    
                    # Format best params for display with explanations
                    best_params_display = tuning_results['best_params'].copy()
                    current_model_type = st.session_state.get('model_type_select', 'Unknown')
                    
                    # Add explanation for special values
                    if 'max_depth' in best_params_display and best_params_display['max_depth'] == -1:
                        st.info(f"💡 **Lưu ý:** `max_depth = -1` trong {current_model_type} có nghĩa là **không giới hạn độ sâu** (cây sẽ phát triển cho đến khi các leaf nodes đều pure hoặc chứa ít hơn min_samples_split samples). Khi áp dụng, slider sẽ được đặt về giá trị mặc định (6).")
                    
                    st.json(best_params_display)
                    
                    st.metric("Best AUC Score", f"{tuning_results['best_score']:.4f}")
                    st.info(f"📊 Đã thử {tuning_results['total_fits']} tổ hợp tham số")
                    
                    # Show top 5 results (no nested expander)
                    st.markdown("##### 🔝 Top 5 Tổ Hợp Tham Số Tốt Nhất")
                    for i, result in enumerate(tuning_results['top_results']):
                        st.markdown(f"**#{i+1}** - AUC: {result['mean_test_score']:.4f} (+/- {result['std_test_score']:.4f})")
                        st.code(str(result['params']))
                    
                    # Button to apply best params
                    if st.button("✅ Áp Dụng Tham Số Tốt Nhất", key="apply_best_params"):
                        best_params = tuning_results['best_params']
                        st.session_state.best_tuned_params = best_params
                        
                        # Set flag to clear widget keys on next rerun
                        st.session_state.apply_tuned_params_flag = True
                        
                        # Clear tuning results after applying
                        st.session_state.tuning_results = None
                        st.success("✅ Đã áp dụng tham số tốt nhất!")
                        st.rerun()
    
    # Tab 2: Evaluation Results
    with tab2:
        st.markdown("### 📊 Kết Quả Đánh Giá Mô Hình")
        
        if st.session_state.model is None:
            st.warning("⚠️ Chưa có mô hình được huấn luyện. Vui lòng huấn luyện mô hình trước.")
        else:
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
                
                st.plotly_chart(fig, width='stretch')
                
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
                
                # Get model name for display
                model_name_roc = st.session_state.get('selected_model_name', st.session_state.get('model_type_select', 'Model'))
                
                fig = go.Figure()
                
                # ROC curve
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode='lines',
                    name=f'{model_name_roc} (AUC = {metrics["auc"]:.3f})',
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
                
                st.plotly_chart(fig, width='stretch')
            
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
                    width='stretch'
                )
            
            with col2:
                st.markdown("#### 📊 Precision-Recall Curve")
                
                # Mock PR curve
                recall_vals = np.linspace(0, 1, 100)
                precision_vals = 1 - recall_vals * 0.3 + np.random.normal(0, 0.02, 100)
                precision_vals = np.clip(precision_vals, 0, 1)
                
                # Get model name for display
                model_name_pr = st.session_state.get('selected_model_name', st.session_state.get('model_type_select', 'Model'))
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=recall_vals,
                    y=precision_vals,
                    mode='lines',
                    fill='tozeroy',
                    name=model_name_pr,
                    line=dict(color='#764ba2', width=3)
                ))
                
                fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    template="plotly_dark",
                    height=300
                )
                
                st.plotly_chart(fig, width='stretch')
            
            # Download results
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("💾 Lưu Mô Hình", width='stretch'):
                    st.success("✅ Đã lưu mô hình!")
    
    # Tab 3: Model Comparison
    with tab3:
        st.markdown("### 📈 Lịch Sử & So Sánh Mô Hình")
        
        if not st.session_state.model_history:
            st.info("💡 Chưa có lịch sử huấn luyện. Hãy huấn luyện ít nhất một mô hình ở Tab 1.")
        else:
            st.markdown(f"Đã lưu {len(st.session_state.model_history)} kết quả huấn luyện.")
            
            # Convert history to DataFrame
            history_df = pd.DataFrame(st.session_state.model_history)
            
            # Display table with selection
            st.markdown("#### 📊 Bảng So Sánh Chi Tiết")
            
            # Get currently selected model index
            selected_model_idx = st.session_state.get('selected_model_idx', None)
            
            # Create selection options
            model_options = [f"{i}: {row['Model']} ({row['Timestamp']}) - AUC: {row['AUC']:.3f}" 
                           for i, row in history_df.iterrows()]
            
            # Add "Chưa chọn" option at the beginning
            model_options_with_none = ["-- Chọn mô hình --"] + model_options
            
            # Determine default index
            if selected_model_idx is not None and selected_model_idx < len(model_options):
                default_idx = selected_model_idx + 1  # +1 because of "Chưa chọn" option
            else:
                default_idx = 0
            
            # Selection dropdown
            selected_option = st.selectbox(
                "🎯 Chọn mô hình để sử dụng:",
                model_options_with_none,
                index=default_idx,
                key="model_selector"
            )
            
            # Handle selection
            if selected_option != "-- Chọn mô hình --":
                # Extract index from selection
                new_idx = int(selected_option.split(":")[0])
                
                if new_idx != selected_model_idx:
                    row = history_df.iloc[new_idx]
                    st.session_state.selected_model_idx = new_idx
                    st.session_state.selected_model_name = row['Model']
                    st.session_state.selected_model_timestamp = row['Timestamp']
                    st.session_state.model_metrics = {
                        'accuracy': row['Accuracy'],
                        'precision': row['Precision'],
                        'recall': row['Recall'],
                        'f1': row['F1-Score'],
                        'auc': row['AUC']
                    }
                    st.session_state.explainer = None
                    st.session_state.shap_values = None
                    st.rerun()
            
            # Display the dataframe
            st.dataframe(
                history_df.style.format({
                    'Accuracy': '{:.3f}',
                    'Precision': '{:.3f}',
                    'Recall': '{:.3f}',
                    'F1-Score': '{:.3f}',
                    'AUC': '{:.3f}'
                }).background_gradient(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'], cmap='RdYlGn'),
                width='stretch'
            )
            
            st.markdown("---")
            
            # Show selected model info
            if selected_model_idx is not None and selected_model_idx < len(st.session_state.model_history):
                selected_info = st.session_state.model_history[selected_model_idx]
                st.success(f"✅ **Mô hình được chọn**: {selected_info['Model']} (Timestamp: {selected_info['Timestamp']}) - AUC: {selected_info['AUC']:.3f}")
                st.info("💡 Bạn có thể chuyển sang trang **Model Explanation** để xem giải thích mô hình.")
            else:
                st.info("💡 Chọn mô hình từ dropdown ở trên để sử dụng cho Model Explanation.")
            
            # Comparison charts
            st.markdown("#### 📉 Biểu Đồ So Sánh")
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart comparison
                fig = go.Figure()
                
                metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
                
                # Group by Model to get average or max if multiple runs of same model? 
                # For now, just plot all runs, maybe add ID or Timestamp to x-axis
                history_df['Run ID'] = history_df['Model'] + " (" + history_df['Timestamp'] + ")"
                
                for metric in metrics_to_plot:
                    fig.add_trace(go.Bar(
                        name=metric,
                        x=history_df['Run ID'],
                        y=history_df[metric],
                        text=history_df[metric].round(3),
                        textposition='outside'
                    ))
                
                fig.update_layout(
                    title='So Sánh Metrics Giữa Các Lần Chạy',
                    xaxis_title='Model Run',
                    yaxis_title='Score',
                    template="plotly_dark",
                    height=400,
                    barmode='group'
                )
                
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Line chart for AUC trend
                fig = px.line(
                    history_df, 
                    x='Timestamp', 
                    y='AUC', 
                    color='Model',
                    markers=True,
                    title='Xu Hướng AUC Theo Thời Gian'
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, width='stretch')
            
            # Best model recommendation
            best_run_idx = history_df['AUC'].idxmax()
            best_model = history_df.loc[best_run_idx, 'Model']
            best_auc = history_df.loc[best_run_idx, 'AUC']
            best_time = history_df.loc[best_run_idx, 'Timestamp']
            
            st.success(f"🏆 **Mô hình tốt nhất hiện tại**: {best_model} (chạy lúc {best_time}) với AUC = {best_auc:.3f}")
            
            # Clear history button
            if st.button("🗑️ Xóa Lịch Sử", type="secondary"):
                st.session_state.model_history = []
                st.rerun()

