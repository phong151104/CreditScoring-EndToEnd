# 🏦 Credit Scoring System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Hệ thống Đánh giá và Dự đoán Điểm Tín dụng** sử dụng Machine Learning, với giao diện web chuyên nghiệp và khả năng giải thích AI (Explainable AI).

> 📌 **Đồ án Tốt nghiệp** - Xây dựng hệ thống Credit Scoring end-to-end từ tiền xử lý dữ liệu đến dự đoán và giải thích quyết định.

---

## 📋 Mục lục

- [Tính năng chính](#-tính-năng-chính)
- [Demo Screenshots](#-demo-screenshots)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Chạy ứng dụng](#-chạy-ứng-dụng)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cấu hình LLM](#-cấu-hình-llm-optional)
- [Dữ liệu mẫu](#-dữ-liệu-mẫu)
- [Đóng góp nổi bật](#-đóng-góp-nổi-bật)
- [Tác giả](#-tác-giả)

---

## ✨ Tính Năng Chính

### 📤 1. Upload & Phân tích Dữ liệu (EDA)
- Upload file CSV dữ liệu khách hàng
- Thống kê mô tả (mean, median, std, quartiles)
- Biểu đồ phân phối: Histogram, Boxplot, Violin Plot
- Phân tích tương quan (Correlation Heatmap)
- Phát hiện Missing Values, Outliers
- **🤖 AI Analysis**: Phân tích tự động bằng Google Gemini

### ⚙️ 2. Feature Engineering (Tiền xử lý dữ liệu)
| Chức năng | Phương pháp hỗ trợ |
|-----------|-------------------|
| **Xử lý Missing Values** | Mean, Median, Mode, Constant, Forward/Backward Fill |
| **Xử lý Outliers** | IQR Method, Z-Score, Winsorization |
| **Mã hóa Categorical** | One-Hot, Label, Target, Ordinal, Frequency Encoding |
| **Chuẩn hóa (Scaling)** | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler |
| **Binning** | Equal-width, Quantile-based, Custom bins |
| **Cân bằng dữ liệu** | SMOTE, ADASYN, Random Under/Over Sampling, SMOTE-ENN, SMOTE-Tomek |
| **Chia dữ liệu** | Train/Validation/Test split với stratification |

> ⚠️ **Đảm bảo không Data Leakage**: Pipeline fit trên Train, transform trên tất cả sets.

### 🤖 3. Huấn luyện Mô hình (Model Training)

**Các thuật toán hỗ trợ:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- **🔥 Stacking Ensemble** (kết hợp nhiều models)

**Tính năng nổi bật:**
- ✅ Hyperparameter Tuning (Grid Search, Random Search)
- ✅ Cross-Validation (K-Fold)
- ✅ Early Stopping cho Boosting models
- ✅ So sánh Train/Validation/Test metrics
- ✅ Phát hiện Overfitting tự động
- ✅ OOF (Out-of-Fold) Tuning cho Stacking

### 📊 4. Đánh giá Mô hình (Evaluation)
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Visualization**: Confusion Matrix, ROC Curve
- **Model Comparison**: Bảng so sánh tất cả models đã train
- **Training History**: Lịch sử các lần huấn luyện

### 🔍 5. Giải thích Mô hình (SHAP Explainability)
- **Global Explanation**: Feature importance tổng thể
- **Local Explanation**: Giải thích từng dự đoán cá nhân
- **Visualizations**: Summary Plot, Beeswarm Plot, Waterfall Plot, Force Plot
- **🤖 AI Interpretation**: Gemini AI giải thích SHAP values bằng ngôn ngữ tự nhiên
- **💬 Q&A Chat**: Hỏi đáp với AI về mô hình

### 🎯 6. Dự đoán & Tư vấn (Prediction & Advisory)
- Form nhập thông tin khách hàng
- **Credit Score**: Điểm tín dụng (300-850) theo công thức chuẩn Basel II/III
- **Risk Classification**: Phân loại rủi ro 5 cấp (Very Low → Very High)
- **Approval Decision**: Phê duyệt / Cần bổ sung / Từ chối
- **🤖 AI Recommendations**: Gợi ý cải thiện điểm tín dụng

---


## 💻 Yêu Cầu Hệ Thống

| Yêu cầu | Phiên bản |
|---------|-----------|
| **Python** | **3.11** (khuyến khích) hoặc 3.10+ |
| **RAM** | 8GB+ (16GB khuyến khích cho datasets lớn) |
| **Disk** | 2GB+ free space |
| **OS** | Windows 10/11, macOS, Linux |

> 💡 **Khuyến khích sử dụng Python 3.11** - Đã test ổn định trên version này.

---

## 🚀 Cài Đặt

### Bước 1: Clone Repository

```bash
git clone <repository-url>
cd demo
```

2. Tạo môi trường ảo với Conda (khuyến nghị):
```bash
# Tạo môi trường mới với Python 3.11+
conda create -n credit-scoring python=3.11

# Kích hoạt môi trường
conda activate credit-scoring
```

> **💡 Lưu ý**: Khuyến khích sử dụng Conda thay vì venv để quản lý môi trường Python, giúp tránh các xung đột dependencies và dễ dàng quản lý phiên bản Python.

3. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình Environment (Optional)

```bash
# Copy file example
cp env.example .env

# Chỉnh sửa .env với API key của bạn (cho tính năng AI)
# GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Chạy Ứng Dụng

### Cách 1: Sử dụng Streamlit CLI

```bash
streamlit run app.py
```

### Cách 2: Sử dụng script có sẵn

**Windows:**
```cmd
RUN_APP.bat
```

**Linux/macOS:**
```bash
chmod +x RUN_APP.sh
./RUN_APP.sh
```

### Truy cập ứng dụng

Mở trình duyệt và truy cập:
```
http://localhost:8501
```

---

## � Hướng Dẫn Sử Dụng

### Workflow cơ bản

```
1. Dashboard           → Xem tổng quan hệ thống
       ↓
2. Data Upload & EDA   → Upload CSV, phân tích dữ liệu
       ↓
3. Feature Engineering → Tiền xử lý, chia Train/Valid/Test
       ↓
4. Model Training      → Huấn luyện và so sánh models
       ↓
5. Model Explanation   → Xem SHAP values, giải thích AI
       ↓
6. Prediction          → Dự đoán cho khách hàng mới
```

### Chi tiết từng bước

#### Bước 1: Upload dữ liệu
1. Vào **Data Upload & Analysis**
2. Upload file CSV (xem mục [Dữ liệu mẫu](#-dữ-liệu-mẫu))
3. Xem thống kê và biểu đồ EDA
4. (Optional) Bấm **"Phân tích bằng AI"** để nhận insights

#### Bước 2: Tiền xử lý dữ liệu
1. Vào **Feature Engineering**
2. Xử lý Missing Values (chọn phương pháp phù hợp)
3. Xử lý Outliers nếu cần
4. Mã hóa biến categorical
5. Chia dữ liệu: Train (70%) / Validation (15%) / Test (15%)
6. Chọn features và target column

#### Bước 3: Huấn luyện mô hình
1. Vào **Model Training**
2. Chọn model (VD: XGBoost, Stacking Ensemble)
3. Cấu hình hyperparameters
4. Bật **Early Stopping** (cho Boosting models)
5. Bấm **"Huấn Luyện Mô Hình"**
6. Xem kết quả Accuracy, AUC, Confusion Matrix

#### Bước 4: Giải thích mô hình
1. Vào **Model Explanation**
2. Bấm **"Khởi Tạo SHAP"**
3. Xem Global Feature Importance
4. Chọn sample để xem Local Explanation
5. (Optional) Chat với AI về mô hình

#### Bước 5: Dự đoán
1. Vào **Prediction & Advisory**
2. Nhập thông tin khách hàng
3. Xem Credit Score, Risk Level, Approval Decision
4. Xem recommendations từ AI

---

## 📁 Cấu Trúc Dự Án

```
credit-scoring-system/
│
├── 📄 app.py                           # Entry point - Streamlit application
├── 📄 requirements.txt                 # Python dependencies
├── 📄 .env                             # Environment variables (API keys)
├── 📄 env.example                      # Template cho .env
├── 📄 RUN_APP.bat                      # Windows startup script
├── 📄 RUN_APP.sh                       # Linux/macOS startup script
│
├── 📂 .streamlit/
│   └── config.toml                     # Streamlit theme & server config
│
├── 📂 views/                           # Frontend - Streamlit pages
│   ├── home.py                         # Dashboard tổng quan
│   ├── upload_eda.py                   # Upload & Exploratory Data Analysis
│   ├── feature_engineering.py          # Tiền xử lý dữ liệu (~4000 lines)
│   ├── model_training.py               # Huấn luyện mô hình (~1600 lines)
│   ├── shap_explanation.py             # SHAP Explainability
│   └── prediction.py                   # Dự đoán & Tư vấn
│
├── 📂 backend/                         # Backend logic
│   │
│   ├── 📂 data_processing/             # Tiền xử lý dữ liệu
│   │   ├── preprocessing_pipeline.py  # Pipeline đảm bảo no data leakage
│   │   ├── encoder.py                  # Categorical encoding (5 methods)
│   │   ├── balancer.py                 # Data balancing (SMOTE, ADASYN, ...)
│   │   └── outlier_handler.py          # Outlier detection & handling
│   │
│   ├── 📂 models/                      # Machine Learning
│   │   ├── trainer.py                  # Model training, Stacking, OOF Tuning
│   │   ├── predictor.py                # Prediction + Credit Score formula
│   │   └── evaluator.py                # Metrics calculation
│   │
│   ├── 📂 explainability/              # Model Explanation
│   │   └── shap_explainer.py           # SHAP implementation
│   │
│   └── 📂 llm_integration/             # AI Integration
│       ├── config.py                   # LLM configuration
│       ├── eda_analyzer.py             # AI EDA analysis (Gemini)
│       └── shap_analyzer.py            # AI SHAP interpretation
│
├── 📂 utils/                           # Utilities
│   ├── session_state.py                # Session management
│   └── ui_components.py                # Reusable UI components
│
├── 📂 sample_data/                     # Sample datasets
│   ├── generate_data.py                # Script tạo dữ liệu giả
│   ├── README.md                       # Hướng dẫn về dữ liệu
│   └── QUICKSTART_DATA.md              # Quick start với dữ liệu mẫu
│
└── 📂 assets/                          # Static assets
    └── logo.png                        # Logo (optional)
```

---

## 🛠 Công Nghệ Sử Dụng

### Core Framework
| Công nghệ | Version | Mô tả |
|-----------|---------|-------|
| **Python** | 3.11 | Ngôn ngữ lập trình |
| **Streamlit** | 1.51.0 | Web framework |

### Data Processing
| Công nghệ | Version | Mô tả |
|-----------|---------|-------|
| **Pandas** | 2.1.4 | Data manipulation |
| **NumPy** | 1.26.2 | Numerical computing |
| **SciPy** | 1.11.4 | Scientific computing |
| **Statsmodels** | 0.14.1 | Statistical analysis |

### Machine Learning
| Công nghệ | Version | Mô tả |
|-----------|---------|-------|
| **Scikit-learn** | 1.5.2 | ML algorithms |
| **XGBoost** | 2.0.3 | Gradient Boosting |
| **LightGBM** | 4.1.0 | Light Gradient Boosting |
| **CatBoost** | 1.2.2 | Categorical Boosting |
| **Imbalanced-learn** | 0.12.4 | Resampling techniques |
| **Optuna** | 3.5.0 | Hyperparameter tuning |

### Explainability
| Công nghệ | Version | Mô tả |
|-----------|---------|-------|
| **SHAP** | 0.44.0 | SHapley Additive exPlanations |

### Visualization
| Công nghệ | Version | Mô tả |
|-----------|---------|-------|
| **Plotly** | 5.18.0 | Interactive charts |
| **Matplotlib** | 3.8.2 | Static charts |
| **Seaborn** | 0.13.0 | Statistical visualization |

### AI Integration
| Công nghệ | Version | Mô tả |
|-----------|---------|-------|
| **Google Generative AI** | 0.7.2 | Gemini API |

---

## 🔑 Cấu Hình LLM (Optional)

Để sử dụng tính năng AI (phân tích EDA, giải thích SHAP, Q&A), bạn cần cấu hình API key.

### Google Gemini (Khuyến khích - Có free tier)

1. Truy cập [Google AI Studio](https://aistudio.google.com/)
2. Tạo API key
3. Thêm vào file `.env`:
```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
LLM_PROVIDER=google
```

### OpenAI GPT (Alternative)

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
LLM_PROVIDER=openai
```

### Anthropic Claude (Alternative)

```env
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229
LLM_PROVIDER=anthropic
```

> 💡 Không có API key? Ứng dụng vẫn hoạt động bình thường, chỉ thiếu tính năng AI analysis.

---

## 📊 Dữ Liệu Mẫu

### Format CSV yêu cầu

File CSV cần có:
- **Features**: Các đặc trưng của khách hàng (age, income, loan_amount, ...)
- **Target**: Cột nhãn (0 = không vỡ nợ, 1 = vỡ nợ)

### Ví dụ

```csv
customer_id,age,income,employment_years,loan_amount,credit_history_length,num_credit_cards,debt_ratio,default
1001,35,50000,5,10000,8,2,0.25,0
1002,42,75000,12,15000,15,3,0.18,0
1003,28,30000,2,5000,3,1,0.45,1
1004,55,120000,25,30000,20,4,0.12,0
```

### Tạo dữ liệu mẫu

```bash
cd sample_data
python generate_data.py
```

Script sẽ tạo file `credit_data.csv` với 1000 records giả lập.

---

## 🏆 Đóng Góp Nổi Bật

### Kỹ thuật
| # | Đóng góp | Mô tả |
|---|----------|-------|
| 1 | **PreprocessingPipeline** | Pipeline tiền xử lý đảm bảo không data leakage |
| 2 | **OOF Tuning cho Stacking** | Hyperparameter tuning cho Stacking không overfitting |
| 3 | **Early Stopping + Validation** | Tự động dừng training khi bắt đầu overfit |
| 4 | **Credit Score Basel II/III** | Công thức log-odds scaling theo chuẩn ngành |
| 5 | **5-Tier Risk Classification** | Phân loại rủi ro 5 cấp độ chuẩn industry |
| 6 | **Multi-model SHAP** | SHAP cho Tree, Linear, và Ensemble models |
| 7 | **LLM-powered Analysis** | AI tự động phân tích EDA và giải thích model |
| 8 | **Fragment Optimization** | Tối ưu Streamlit performance |

### Giao diện
- ✅ Dark theme chuyên nghiệp
- ✅ Interactive Plotly charts
- ✅ Responsive layout
- ✅ Bilingual (Tiếng Việt)

---

## 🎨 Tùy Chỉnh

### Theme
Chỉnh sửa `.streamlit/config.toml`:
```toml
[theme]
base = "dark"
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
```

### Logo
Thêm logo vào `assets/logo.png` (kích thước: 400x100px).

---

## 📝 Troubleshooting

### Lỗi thường gặp

**1. ModuleNotFoundError**
```bash
pip install -r requirements.txt --upgrade
```

**2. SHAP chạy chậm**
- Giảm số lượng samples (mặc định: 500 samples)
- Sử dụng TreeExplainer cho tree-based models

**3. LightGBM installation error (Windows)**
```bash
pip install lightgbm --install-option=--nomp
```

**4. CatBoost không tương thích**
```bash
pip install catboost==1.2.2
```

---

## 📄 License

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 👨‍💻 Tác Giả

**Phạm Hùng Phong**
- 📧 Email: phamhungphong1511@gmail.com
- � GitHub: [github.com/phong151104](https://github.com/phong151104)

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Web framework
- [SHAP](https://github.com/shap/shap) - Model explainability
- [Google Gemini](https://ai.google.dev/) - AI integration
- [Scikit-learn](https://scikit-learn.org/) - ML algorithms

---

**Version**: 2.0.0  
**Last Updated**: 2026-01-14

---

<p align="center">
  Made with ❤️ for Credit Scoring
</p>
