"""
UI Components - Các thành phần giao diện tùy chỉnh
"""

import streamlit as st

def load_custom_css():
    """Load custom CSS cho ứng dụng"""
    st.markdown("""
    <style>
        /* Hide Streamlit default menu and header */
        #MainMenu {visibility: hidden !important;}
        header {visibility: hidden !important;}
        footer {visibility: hidden !important;}
        
        /* Hide the top menu bar completely */
        section[data-testid="stSidebarNav"] {display: none !important;}
        
        /* Hide hamburger menu */
        button[kind="header"] {display: none !important;}
        
        /* Main container */
        .main {
            padding: 2rem;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #1f2937 0%, #345f9c 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }
        
        .main-header p {
            color: rgba(255, 255, 255, 0.9);
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }
        
        /* Card styling */
        .info-card {
            background-color: #262730;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .metric-card h3 {
            color: white;
            font-size: 2rem;
            margin: 0;
        }
        
        .metric-card p {
            color: rgba(255, 255, 255, 0.8);
            margin: 0.5rem 0 0 0;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #1f2937 0%, #345f9c 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1a1d24;
        }
        
        [data-testid="stSidebar"] .stRadio > label {
            font-size: 0.95rem;
            font-weight: 500;
            padding: 0.5rem 0;
            letter-spacing: 0.3px;
        }
        
        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.3rem;
        }
        
        [data-testid="stSidebar"] .stRadio label {
            padding: 0.6rem 1rem;
            border-radius: 6px;
            transition: all 0.2s ease;
        }
        
        [data-testid="stSidebar"] .stRadio label:hover {
            background-color: rgba(102, 126, 234, 0.1);
        }
        
        /* Sidebar headers */
        [data-testid="stSidebar"] h3 {
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 1px;
            color: #667eea;
            margin-bottom: 1rem;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #262730;
            border-radius: 5px;
            font-weight: 600;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #262730;
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1f2937 0%, #345f9c 100%);
        }
        
        /* DataFrame styling */
        .dataframe {
            border-radius: 5px;
            overflow: hidden;
        }
        
        /* Alert styling */
        .stAlert {
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background-color: #262730;
            border-radius: 10px;
            padding: 2rem;
            border: 2px dashed #667eea;
        }
        
        /* Metric container */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
        }
        
        /* Plotly chart container */
        .js-plotly-plot {
            border-radius: 10px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render header chính của ứng dụng"""
    st.markdown("""
    <div class="main-header">
        <h1>CREDIT SCORING SYSTEM</h1>
        <p>Advanced Risk Assessment & Prediction Platform</p>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(title, value, delta=None):
    """Render card hiển thị metric"""
    delta_html = f"<p style='color: {'#00ff00' if delta and delta > 0 else '#ff0000'};'>{'▲' if delta and delta > 0 else '▼'} {abs(delta) if delta else ''}</p>" if delta else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>{value}</h3>
        <p>{title}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_info_card(title, content):
    """Render card thông tin"""
    st.markdown(f"""
    <div class="info-card">
        <h4 style="margin-top: 0; color: #667eea;">{title}</h4>
        <p style="margin-bottom: 0;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

def show_llm_analysis(prompt_text, placeholder_response):
    """Hiển thị khu vực phân tích LLM với placeholder"""
    with st.expander("🤖 Phân Tích Tự Động (AI Analysis)", expanded=True):
        st.markdown(f"""
        <div class="info-card">
            <h4 style="margin-top: 0; color: #667eea;">💬 Nhận Xét Từ AI</h4>
            <p style="margin-bottom: 0; font-style: italic;">{placeholder_response}</p>
        </div>
        """, unsafe_allow_html=True)

def show_processing_placeholder(step_name):
    """Hiển thị placeholder cho xử lý backend"""
    st.info(f"🔧 **Backend Placeholder**: {step_name} sẽ được triển khai sau")

