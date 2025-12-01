"""
Dashboard - Home Page
"""

import streamlit as st
from utils.ui_components import render_info_card
from utils.session_state import init_session_state, get_session_info

def render():
    """Render dashboard page"""
    init_session_state()
    
    # Welcome section
    st.markdown("## Welcome to Credit Scoring System")
    
    st.markdown("""
    A comprehensive platform for analyzing, evaluating, and predicting customer creditworthiness 
    using advanced Machine Learning algorithms and explainable AI.
    """)
    
    st.markdown("---")
    
    # Features overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Features")
        
        features = [
            ("↑", "Data Upload & Analysis", "Import and explore your datasets"),
            ("⚡", "Feature Engineering", "Preprocess and select features"),
            ("◈", "Model Training", "Train multiple ML algorithms"),
            ("◐", "Model Explanation", "Understand model decisions with SHAP"),
            ("◎", "Prediction & Advisory", "Generate predictions and recommendations"),
        ]
        
        for icon, title, desc in features:
            st.markdown(f"""
            <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <h4 style="margin: 0; color: #667eea;">{icon} {title}</h4>
                <p style="margin: 0.5rem 0 0 0; color: #aaa;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Current Status")
        
        session_info = get_session_info()
        
        # Data status
        if session_info['has_data']:
            st.success("● Data uploaded successfully")
        else:
            st.warning("○ No data uploaded - Please upload from 'Data Upload & Analysis'")
        
        # Processed data status
        if session_info['has_processed_data']:
            st.success("● Data processed")
        else:
            st.info("○ Data not processed yet")
        
        # Model status
        if session_info['has_model']:
            st.success(f"● Model trained ({st.session_state.model_type})")
        else:
            st.info("○ No model trained")
        
        # Features status
        if session_info['num_features'] > 0:
            st.success(f"● Selected {session_info['num_features']} features")
        else:
            st.info("○ No features selected")
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### ▶ Quick Start Guide")
    
    cols = st.columns(5)
    
    steps = [
        ("①", "Upload\nData"),
        ("②", "Explore &\nAnalyze"),
        ("③", "Process\nFeatures"),
        ("④", "Train\nModel"),
        ("⑤", "Predict &\nExplain"),
    ]
    
    for col, (num, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f2937 0%, #345f9c 100%); 
                        padding: 1.5rem; border-radius: 10px; text-align: center; height: 120px;
                        display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{num}</div>
                <div style="font-size: 0.9rem; font-weight: 600; white-space: pre-line;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technical info
    with st.expander("▼ Technical Information"):
        st.markdown("""
        **Technology Stack:**
        - **Frontend**: Streamlit with Custom Dark Theme
        - **ML Models**: Logistic Regression, XGBoost, LightGBM, CatBoost
        - **Model Explainability**: SHAP (SHapley Additive exPlanations)
        - **AI Analysis**: LLM Integration (OpenAI/LangChain)
        - **Visualization**: Plotly, Matplotlib, Seaborn
        
        **Features:**
        - ▪ Professional, enterprise-grade interface
        - ▪ Optimized dark theme for extended use
        - ▪ Interactive and responsive visualizations
        - ▪ Seamless state management across pages
        - ▪ High performance, scalable architecture
        """)
    
    # Action button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶ Get Started - Upload Data", width='stretch', type="primary"):
            st.info("← Please select '↑ Data Upload & Analysis' from the sidebar to begin!")

