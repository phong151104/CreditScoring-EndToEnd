# TỔNG HỢP ĐỒ ÁN: HỆ THỐNG CREDIT SCORING
## Tài liệu cho Chương 5 - Giải Pháp và Đóng Góp Nổi Bật

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1. Mục tiêu
Xây dựng **hệ thống chấm điểm tín dụng (Credit Scoring System)** hoàn chỉnh, từ xử lý dữ liệu thô đến đưa ra quyết định phê duyệt khoản vay, với khả năng:
- Xử lý và làm sạch dữ liệu tự động
- Huấn luyện và so sánh nhiều mô hình Machine Learning
- Giải thích quyết định của mô hình (Explainable AI)
- Tích hợp AI (LLM) để phân tích và hỗ trợ người dùng

### 1.2. Công nghệ sử dụng
| Thành phần | Công nghệ |
|------------|-----------|
| Frontend | Streamlit (Python Web Framework) |
| Backend | Python 3.10+ |
| ML Framework | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Data Balancing | imbalanced-learn (SMOTE, ADASYN, etc.) |
| LLM Integration | Google Gemini AI (gemini-2.5-flash) |
| Visualization | Plotly, Matplotlib |

### 1.3. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT FRONTEND                       │
│  ┌──────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        │
│  │Dashboard │ │Data Upload│ │  Feature  │ │  Model    │        │
│  │          │ │   & EDA   │ │Engineering│ │ Training  │        │
│  └──────────┘ └───────────┘ └───────────┘ └───────────┘        │
│  ┌───────────────────────────┐ ┌───────────────────────┐        │
│  │     SHAP Explanation      │ │ Prediction & Advisory │        │
│  └───────────────────────────┘ └───────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          BACKEND                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Data Processing │  │     Models      │  │  Explainability │  │
│  │  - Pipeline     │  │  - Trainer      │  │  - SHAP Explainer│ │
│  │  - Encoder      │  │  - Predictor    │  │                 │  │
│  │  - Balancer     │  │  - Evaluator    │  │                 │  │
│  │  - Outlier      │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    LLM Integration                          ││
│  │  - EDA Analyzer (phân tích dữ liệu bằng AI)                ││
│  │  - SHAP Analyzer (giải thích model bằng AI)                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. CÁC GIẢI PHÁP KỸ THUẬT NỔI BẬT

### 2.1. Pipeline Tiền Xử Lý Dữ Liệu (PreprocessingPipeline)

**Vấn đề cần giải quyết**: Đảm bảo không xảy ra **Data Leakage** - tình trạng thông tin từ tập test/validation "rò rỉ" vào quá trình training.

**Giải pháp triển khai**:
Xây dựng class `PreprocessingPipeline` với nguyên tắc **Fit on Train, Transform on All**:

```python
class PreprocessingPipeline:
    """
    Đảm bảo fit trên train data, transform trên tất cả datasets.
    Lưu trữ các fitted transformers để tái sử dụng.
    """
    
    def fit_imputer(self, train_data, column, method):
        """Fit imputer CHỈ trên train data"""
        if method == "Mean":
            fill_value = train_data[column].mean()
        elif method == "Median":
            fill_value = train_data[column].median()
        # Lưu giá trị để transform cho valid/test
        self.imputers[column] = {'method': method, 'fill_value': fill_value}
    
    def transform_imputation(self, data, column):
        """Apply imputation đã fit lên bất kỳ dataset nào"""
        fill_value = self.imputers[column]['fill_value']
        data[column] = data[column].fillna(fill_value)
        return data
```

**Các chức năng được triển khai**:

| Chức năng | Phương pháp hỗ trợ |
|-----------|-------------------|
| **Xử lý Missing Values** | Mean, Median, Mode, Constant, Forward/Backward Fill |
| **Outlier Handling** | IQR Method, Z-Score, Winsorization |
| **Categorical Encoding** | One-Hot, Label, Target, Ordinal, Frequency Encoding |
| **Feature Scaling** | StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler |
| **Data Validation** | Kiểm tra giá trị âm, ngưỡng min/max, khoảng giá trị |

---

### 2.2. Xử Lý Dữ Liệu Mất Cân Bằng (Data Balancing)

**Vấn đề**: Trong credit scoring, tỷ lệ default thường rất thấp (1-5%), gây ra **Class Imbalance** làm model thiên về class majority.

**Giải pháp triển khai**:
Module `balancer.py` hỗ trợ nhiều kỹ thuật resampling:

```python
def balance_data(data, target_column, method="SMOTE"):
    """
    Hỗ trợ các phương pháp:
    - SMOTE: Synthetic Minority Over-sampling Technique
    - ADASYN: Adaptive Synthetic Sampling
    - Random Undersampling
    - Random Oversampling
    - SMOTE-ENN: Kết hợp SMOTE + Edited Nearest Neighbors
    - SMOTE-Tomek: Kết hợp SMOTE + Tomek Links
    """
```

**Đóng góp nổi bật**:
- Tự động phát hiện imbalance ratio và gợi ý phương pháp phù hợp
- Hiển thị visualization so sánh before/after balancing
- Lưu trữ thông tin balancing để reproductibility

---

### 2.3. Mô Hình Stacking Ensemble với OOF (Out-of-Fold) Tuning

**Vấn đề**: Hyperparameter tuning cho Stacking dễ gây overfitting nếu dùng chung data cho cả tuning và meta-model training.

**Giải pháp triển khai**:
Áp dụng **OOF Predictions** để tránh data leakage:

```python
def tune_stacking_with_oof(X_train, y_train, X_test, y_test, params, ...):
    """
    STEP 1: Tune từng Base Model bằng GridSearchCV/RandomizedSearchCV
            với K-Fold Cross-Validation
    
    STEP 2: Tạo OOF Predictions
            - Với mỗi fold, train base models trên K-1 folds
            - Dự đoán trên fold còn lại
            - Kết quả: ma trận [n_samples, n_base_models]
    
    STEP 3: Tune Meta Model
            - Sử dụng OOF predictions làm features
            - Meta model chỉ "thấy" predictions, không thấy raw features
    
    STEP 4: Build Final Stacking Model với best params
    """
```

**Các Base Models hỗ trợ**:
- Logistic Regression (LR)
- Decision Tree (DT)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Gradient Boosting (GB)

**Các Meta Models hỗ trợ**:
- Random Forest
- Logistic Regression
- XGBoost

---

### 2.4. Early Stopping cho Boosting Models

**Vấn đề**: Boosting models (XGBoost, LightGBM, CatBoost) dễ overfitting nếu train quá nhiều iterations.

**Giải pháp triển khai**:
Sử dụng **Validation Set** riêng biệt để monitoring và early stopping:

```python
def train_model(X_train, y_train, X_test, y_test, model_type, params,
                X_valid=None, y_valid=None, early_stopping_rounds=None):
    """
    - Train set: Dùng để fit model
    - Validation set: Dùng cho Early Stopping (KHÔNG phải test)
    - Test set: CHỈ dùng để evaluate cuối cùng
    """
    
    if model_type == "XGBoost":
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],  # Monitor trên validation
                  early_stopping_rounds=early_stopping_rounds,
                  verbose=False)
```

**Tính năng bổ sung**:
- Hiển thị bảng so sánh **Train vs Validation vs Test** metrics
- Tự động cảnh báo khi phát hiện **Overfitting** (Train >> Validation)
- Thông báo iteration mà model stopped

---

### 2.5. Công Thức Tính Credit Score chuẩn Industry

**Vấn đề**: Chuyển đổi xác suất default (PD) thành credit score theo chuẩn ngành tài chính.

**Giải pháp triển khai**:
Sử dụng công thức **Log-Odds Scaling** chuẩn Basel II/III:

```python
def calculate_credit_score(prob_default):
    """
    Công thức: Score = BaseScore + Factor × ln(Odds / BaseOdds)
    
    Các tham số chuẩn industry:
    - PDO (Points to Double Odds) = 30
    - Base Score = 600 (tại odds 19:1, tức PD = 5%)
    - Base Odds = 19
    """
    pdo = 30
    base_score = 600
    base_odds = 19
    factor = pdo / math.log(2)  # ≈ 43.29
    
    odds = (1 - prob_default) / prob_default
    credit_score = base_score + factor * math.log(odds / base_odds)
    
    return max(300, min(850, int(credit_score)))  # Clamp to valid range
```

**Phân loại rủi ro 5 cấp** (chuẩn industry):

| PD Range | Risk Level | Credit Score |
|----------|------------|--------------|
| < 2% | Very Low (Rất thấp) | ≥ 750 |
| 2% - 5% | Low (Thấp) | 650 - 749 |
| 5% - 10% | Medium (Trung bình) | 550 - 649 |
| 10% - 20% | High (Cao) | 450 - 549 |
| > 20% | Very High (Rất cao) | < 450 |

---

### 2.6. SHAP Explainability (Giải Thích Mô Hình)

**Vấn đề**: Mô hình Machine Learning thường là "black box", khó giải thích cho stakeholders.

**Giải pháp triển khai**:
Tích hợp **SHAP (SHapley Additive exPlanations)**:

```python
class SHAPExplainer:
    """
    Hỗ trợ nhiều loại explainer:
    - TreeExplainer: Cho Random Forest, XGBoost, LightGBM, CatBoost
    - LinearExplainer: Cho Logistic Regression
    - KernelExplainer: Fallback cho bất kỳ model nào (bao gồm Stacking)
    """
    
    def compute_shap_values(self, X):
        """Tính SHAP values cho từng sample"""
        
    def get_feature_importance(self):
        """Global feature importance = mean(|SHAP|)"""
        
    def get_local_explanation(self, sample_idx):
        """Local explanation cho 1 sample cụ thể"""
```

**Các visualization được triển khai**:
- **Summary Plot**: Tổng quan feature importance
- **Beeswarm Plot**: Phân bố SHAP values
- **Waterfall Plot**: Giải thích từng dự đoán cá nhân
- **Force Plot**: Trực quan hóa contribution

---

### 2.7. Tích Hợp LLM (Large Language Model)

**Vấn đề**: Người dùng không chuyên về ML cần hỗ trợ diễn giải kết quả.

**Giải pháp triển khai**:
Tích hợp **Google Gemini AI** cho 2 chức năng:

#### a) EDA Analyzer
```python
class LLMEDAAnalyzer:
    """
    - Thu thập toàn bộ thống kê từ EDA (missing, outliers, correlations, ...)
    - Gửi context đến Gemini AI
    - Nhận phân tích chi tiết dạng Markdown
    """
    
    def analyze(self, data: pd.DataFrame):
        prompt = self.create_analysis_prompt(eda_summary)
        analysis = self._call_llm(prompt)
        return analysis  # Markdown format
```

#### b) SHAP Analyzer
```python
class SHAPAnalyzer:
    """
    - Giải thích Global SHAP (tổng quan model)
    - Giải thích Local SHAP (từng dự đoán)
    - Hỗ trợ Chat Q&A về model
    """
    
    def chat(self, user_question, model_context, conversation_history):
        """Interactive chat với AI về mô hình"""
```

---

### 2.8. Streamlit Fragment Optimization

**Vấn đề**: Streamlit mặc định rerun toàn bộ script khi có interaction, gây chậm với form dài.

**Giải pháp triển khai**:
Sử dụng **@st.fragment** decorator để cô lập các component:

```python
@st.fragment
def missing_values_fragment(data, missing_data):
    """
    Fragment này CHỈ rerun khi tương tác bên trong,
    KHÔNG làm rerun toàn bộ trang feature_engineering.
    """
    if st.button("Xử lý Missing"):
        # Chỉ rerun phần này, không rerun các fragment khác
        st.rerun(scope="fragment")
```

**Các fragment được triển khai**:
- `remove_columns_fragment`: Loại bỏ cột
- `validation_fragment`: Kiểm tra giá trị
- `outliers_transform_fragment`: Xử lý outliers
- `missing_values_fragment`: Xử lý missing values
- `encoding_fragment`: Mã hóa categorical
- `binning_fragment`: Phân nhóm biến
- `scaling_fragment`: Chuẩn hóa
- `balancing_fragment`: Cân bằng dữ liệu
- `feature_selection_fragment`: Chọn features

---

## 3. TỔNG HỢP CÁC ĐÓNG GÓP NỔI BẬT

### 3.1. Về Mặt Kỹ Thuật

| STT | Đóng góp | Mô tả |
|-----|----------|-------|
| 1 | **PreprocessingPipeline** | Pipeline tiền xử lý đảm bảo không data leakage, fit on train - transform on all |
| 2 | **OOF Tuning cho Stacking** | Hyperparameter tuning cho Stacking Ensemble không bị overfitting |
| 3 | **Early Stopping với Validation Set** | Tự động dừng training khi model bắt đầu overfit |
| 4 | **Train/Valid/Test Metrics Comparison** | Phát hiện overfitting sớm qua so sánh metrics 3 tập |
| 5 | **Credit Score Formula chuẩn Basel** | Công thức log-odds scaling theo tiêu chuẩn ngành |
| 6 | **5-Tier Risk Classification** | Phân loại rủi ro 5 cấp độ chuẩn industry |
| 7 | **Multi-model SHAP Support** | SHAP cho cả Tree-based, Linear, và Ensemble models |
| 8 | **LLM-powered Analysis** | AI tự động phân tích EDA và giải thích model |
| 9 | **Fragment-based UI** | Tối ưu performance Streamlit với fragments |

### 3.2. Về Mặt Ứng Dụng

| STT | Đóng góp | Mô tả |
|-----|----------|-------|
| 1 | **End-to-End System** | Từ upload data → preprocessing → training → prediction → explanation |
| 2 | **Interactive Feature Engineering** | UI trực quan cho từng bước tiền xử lý |
| 3 | **Model Comparison Dashboard** | So sánh nhiều models cùng lúc |
| 4 | **Explainable Predictions** | Mỗi dự đoán đi kèm giải thích "tại sao" |
| 5 | **Actionable Recommendations** | Gợi ý cải thiện điểm tín dụng cho khách hàng |
| 6 | **Bilingual Support** | Giao diện tiếng Việt, output tiếng Việt |

### 3.3. Về Mặt Nghiên Cứu

| STT | Đóng góp | Mô tả |
|-----|----------|-------|
| 1 | **Stacking Generalization** | Triển khai đầy đủ phương pháp từ paper "A Stacking Generalization Approach" |
| 2 | **Benchmark 8+ Models** | So sánh LR, DT, SVM, KNN, RF, GB, XGBoost, LightGBM, CatBoost, Stacking |
| 3 | **Imbalanced Learning** | Đánh giá multiple resampling techniques trên credit data |
| 4 | **XAI in Finance** | Áp dụng SHAP cho giải thích quyết định credit |

---

## 4. SƠ ĐỒ LUỒNG XỬ LÝ

```
┌──────────────────┐
│  Raw Data (CSV)  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Data Upload &   │  • Statistics, Data Types
│  Exploratory     │  • Missing Analysis
│  Data Analysis   │  • Distribution Plots
│                  │  • LLM Analysis
└────────┬─────────┘
         ▼
┌──────────────────┐
│    Feature       │  • Validation
│    Engineering   │  • Missing Imputation
│                  │  • Outlier Handling
│                  │  • Encoding
│                  │  • Scaling
│                  │  • Balancing
│                  │  • Train/Valid/Test Split
└────────┬─────────┘
         ▼
┌──────────────────┐
│     Model        │  • Single Models (LR, RF, XGB, ...)
│    Training      │  • Stacking Ensemble
│                  │  • Hyperparameter Tuning
│                  │  • Early Stopping
│                  │  • Cross-Validation
└────────┬─────────┘
         ▼
┌──────────────────────────────────────────────┐
│              Model Evaluation                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │
│  │Accuracy │ │AUC-ROC  │ │Confusion Matrix │ │
│  │Precision│ │F1-Score │ │ ROC Curve       │ │
│  │Recall   │ │         │ │                 │ │
│  └─────────┘ └─────────┘ └─────────────────┘ │
└────────┬─────────────────────────────────────┘
         ▼
┌──────────────────┐
│     SHAP         │  • Global Explanation
│   Explanation    │  • Local Explanation
│                  │  • Feature Importance
│                  │  • LLM Interpretation
└────────┬─────────┘
         ▼
┌──────────────────┐
│   Prediction &   │  • Credit Score
│    Advisory      │  • Risk Classification
│                  │  • Approval Decision
│                  │  • Recommendations
└──────────────────┘
```

---

## 5. METRICS & EVALUATION

### 5.1. Các Metrics được sử dụng

| Metric | Công thức | Ý nghĩa trong Credit Scoring |
|--------|-----------|------------------------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Tổng quan độ chính xác |
| **Precision** | TP/(TP+FP) | Trong số được dự đoán default, bao nhiêu thực sự default |
| **Recall** | TP/(TP+FN) | Trong số thực sự default, bao nhiêu được phát hiện |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean của Precision và Recall |
| **AUC-ROC** | Area under ROC curve | Khả năng phân biệt good vs bad |

### 5.2. Visualization

- **Confusion Matrix**: Ma trận nhầm lẫn TP/TN/FP/FN
- **ROC Curve**: Đường cong ROC với AUC
- **Model Comparison Table**: Bảng so sánh tất cả models
- **Training History**: Lịch sử các lần training

---

## 6. CẤU TRÚC DỰ ÁN

```
credit-scoring-system/
├── app.py                          # Main entry point
├── requirements.txt                # Dependencies
│
├── backend/
│   ├── data_processing/
│   │   ├── preprocessing_pipeline.py  # Pipeline đảm bảo no data leakage
│   │   ├── encoder.py                  # Categorical encoding (5 methods)
│   │   ├── balancer.py                 # Data balancing (6 methods)
│   │   └── outlier_handler.py          # Outlier detection & handling
│   │
│   ├── models/
│   │   ├── trainer.py                  # Model training + Stacking + OOF
│   │   ├── predictor.py                # Prediction + Credit Score
│   │   └── evaluator.py                # Metrics calculation
│   │
│   ├── explainability/
│   │   └── shap_explainer.py           # SHAP implementation
│   │
│   └── llm_integration/
│       ├── eda_analyzer.py             # LLM EDA analysis
│       └── shap_analyzer.py            # LLM SHAP interpretation
│
├── views/
│   ├── home.py                         # Dashboard
│   ├── upload_eda.py                   # Data upload & EDA
│   ├── feature_engineering.py          # All preprocessing steps
│   ├── model_training.py               # Training interface
│   ├── shap_explanation.py             # SHAP visualization
│   └── prediction.py                   # Prediction interface
│
└── utils/
    ├── session_state.py                # Session management
    └── ui_components.py                # Reusable UI components
```

---

## 7. HƯỚNG DẪN SỬ DỤNG (QUICK START)

```bash
# 1. Cài đặt dependencies
pip install -r requirements.txt

# 2. Cấu hình API key (optional, cho LLM features)
cp env.example .env
# Edit .env: GOOGLE_API_KEY=your_key

# 3. Chạy ứng dụng
streamlit run app.py
```

**Workflow cơ bản**:
1. Upload file CSV tại **Data Upload & Analysis**
2. Thực hiện tiền xử lý tại **Feature Engineering**
3. Huấn luyện model tại **Model Training**
4. Xem giải thích tại **Model Explanation**
5. Dự đoán mới tại **Prediction & Advisory**

---

*Tài liệu này được tạo tự động dựa trên phân tích source code của dự án.*
