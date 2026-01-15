# 🏦 Credit Scoring System

Hệ thống đánh giá và dự đoán điểm tín dụng sử dụng Machine Learning với giao diện web chuyên nghiệp.

## ✨ Tính Năng Chính

### 📤 Upload & EDA
- Upload file CSV dữ liệu khách hàng
- Hiển thị bảng dữ liệu mẫu và thống kê mô tả
- Vẽ biểu đồ phân phối (histogram, boxplot, correlation heatmap)
- Phân tích tự động bằng AI (LLM integration)

### ⚙️ Xử Lý & Chọn Biến
- Xử lý giá trị thiếu (imputation, drop)
- Mã hóa biến phân loại (one-hot, label, target encoding)
- Binning cho biến liên tục
- Feature importance analysis
- Lựa chọn biến đầu vào (manual/auto selection)

### 🤖 Huấn Luyện Mô Hình
- Hỗ trợ nhiều thuật toán: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- Điều chỉnh hyperparameters
- Đánh giá mô hình: ROC curve, AUC, Confusion Matrix, Precision, Recall, F1
- So sánh nhiều mô hình

### 🔍 Giải Thích SHAP
- Global explanation (feature importance tổng thể)
- Local explanation (giải thích từng mẫu)
- SHAP waterfall plot, force plot, summary plot
- AI interpretation của SHAP values

### 🎯 Dự Đoán & Gợi Ý
- Form nhập thông tin khách hàng
- Dự đoán điểm tín dụng (300-850 scale)
- Hiển thị mức độ rủi ro
- Giải thích yếu tố ảnh hưởng
- Gợi ý cải thiện điểm tín dụng từ AI

## 🚀 Cài Đặt

### Yêu cầu hệ thống
- Python 3.8+
- pip

### Các bước cài đặt

1. Clone repository:
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

3. (Optional) Tạo file `.env` cho API keys:
```bash
GOOGLE_API_KEY=your_api_key_here
```

## 💻 Chạy Ứng Dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

## 📁 Cấu Trúc Dự Án

```
demo/
├── app.py                      # File chính của ứng dụng
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── .streamlit/
│   └── config.toml            # Cấu hình theme và server
│
├── assets/
│   └── logo.png               # Logo ngân hàng (optional)
│
├── pages/                     # Các trang của ứng dụng
│   ├── __init__.py
│   ├── home.py               # Trang chủ
│   ├── upload_eda.py         # Upload & EDA
│   ├── feature_engineering.py # Xử lý & chọn biến
│   ├── model_training.py     # Huấn luyện mô hình
│   ├── shap_explanation.py   # Giải thích SHAP
│   └── prediction.py         # Dự đoán & gợi ý
│
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── ui_components.py      # UI components tùy chỉnh
│   └── session_state.py      # Quản lý session state
│
└── backend/                   # Backend logic (sẽ phát triển sau)
    ├── data_processing/      # Xử lý dữ liệu
    ├── models/              # ML models
    ├── explainability/      # SHAP & interpretability
    └── llm_integration/     # LLM APIs
```

## 🎨 Giao Diện

- **Theme**: Dark mode chuyên nghiệp
- **Layout**: Wide layout với sidebar navigation
- **Colors**: Gradient purple-blue (#667eea, #764ba2)
- **Charts**: Interactive Plotly charts
- **Responsive**: Tối ưu cho nhiều kích thước màn hình

## 🔧 Tùy Chỉnh

### Theme
Chỉnh sửa file `.streamlit/config.toml` để thay đổi màu sắc và theme.

### Logo
Thêm logo ngân hàng vào `assets/logo.png` (kích thước khuyến nghị: 400x100px).

### Components
Tùy chỉnh UI components trong `utils/ui_components.py`.

## 📊 Dữ Liệu Mẫu

File CSV cần có các cột:
- Các đặc trưng (features): age, income, loan_amount, credit_history, v.v.
- Nhãn (target): default (0 = không vỡ nợ, 1 = vỡ nợ)

Ví dụ:
```csv
customer_id,age,income,credit_history,loan_amount,default
1001,35,50000,good,10000,0
1002,42,75000,excellent,15000,0
1003,28,30000,poor,5000,1
```

## 🚧 Phát Triển Tiếp

### Backend (Cần triển khai)
- [ ] Data preprocessing logic
- [ ] ML model training & evaluation
- [ ] SHAP explainer implementation
- [ ] LLM integration (OpenAI/Claude)
- [ ] Model persistence (save/load)
- [ ] Database integration

### Features mở rộng
- [ ] Multi-user support
- [ ] Model versioning
- [ ] A/B testing
- [ ] Real-time predictions API
- [ ] Batch prediction
- [ ] Model monitoring & drift detection

## 📝 Ghi Chú

- Hiện tại là **giao diện hoàn chỉnh** với mock data
- Backend logic sẽ được triển khai trong giai đoạn tiếp theo
- Tất cả placeholder đều có comment `show_processing_placeholder()`
- Session state đã được setup để duy trì dữ liệu giữa các trang

## 🤝 Đóng Góp

Dự án này đang trong giai đoạn phát triển. Mọi đóng góp đều được chào đón!

## 📄 License

[Thêm license information]

## 👨‍💻 Tác Giả

[Thêm thông tin tác giả]

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-11

