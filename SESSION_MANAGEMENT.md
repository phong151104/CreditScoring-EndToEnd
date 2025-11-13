# 🔄 Session Management - Quản Lý Phiên Làm Việc

## 📌 Tổng Quan

Hệ thống sử dụng **Streamlit Session State** để lưu trữ tất cả cấu hình và dữ liệu trong suốt phiên làm việc. Điều này cho phép bạn:

- ✅ Chuyển đổi giữa các trang mà **KHÔNG MẤT** cấu hình
- ✅ Làm việc với nhiều bước xử lý dữ liệu
- ✅ Quay lại xem lại các bước trước đó
- ✅ Chỉ xóa dữ liệu khi **CHỦ ĐỘNG** muốn

## 🎯 Các Loại Session State

### 1. **Data States**
```python
st.session_state.data              # Dữ liệu gốc từ CSV
st.session_state.processed_data    # Dữ liệu đã xử lý
st.session_state.current_file_id   # ID file hiện tại (tránh reload)
```

### 2. **Configuration States** (GIỮ KHI CHUYỂN TRANG)
```python
st.session_state.missing_config    # Cấu hình xử lý missing data
st.session_state.encoding_config   # Cấu hình mã hóa categorical
st.session_state.scaling_config    # Cấu hình scaling
st.session_state.outlier_config    # Cấu hình xử lý outliers
st.session_state.binning_config    # Cấu hình binning
```

### 3. **Model States**
```python
st.session_state.model             # Model đã train
st.session_state.model_type        # Loại model
st.session_state.model_metrics     # Metrics đánh giá
st.session_state.selected_features # Features đã chọn
```

### 4. **Analysis States**
```python
st.session_state.explainer         # SHAP explainer
st.session_state.shap_values       # SHAP values
st.session_state.ai_analysis       # Phân tích từ LLM
st.session_state.eda_summary       # EDA summary cache
```

## 🔄 Luồng Hoạt Động

### **Upload Data**
```
Upload file → Check if NEW file → 
  ├─ YES: Clear ALL states + Load new data
  └─ NO:  Keep ALL configs + Just update display
```

### **Feature Engineering**
```
Configure → Save to session_state → Apply →
  ├─ Update processed_data
  └─ KEEP configuration (không xóa)
```

### **Page Navigation**
```
Change page → Session state PRESERVED →
  └─ All configs still available
```

### **Manual Clear**
```
Click "Xóa Dữ Liệu" → Clear ALL states →
  └─ Ready for new dataset
```

## 🎨 UI Indicators

### **Sidebar - Session Status**
- ● Data loaded: XXX rows
- ● Model trained
- ● X cấu hình đã lưu
  - Missing: X
  - Encoding: X
  - Binning: X

### **Feature Engineering Page**
- 📋 X cấu hình đã lưu
- 🗑️ Xóa Tất Cả Cấu Hình (button)

### **Upload Page**
- 🗑️ Xóa Dữ Liệu Hiện Tại (when data exists but no file uploaded)

## 💡 Best Practices

### ✅ **DO**
- Upload file một lần và làm việc liên tục
- Sử dụng nhiều tabs/pages thoải mái
- Lưu cấu hình cho từng bước trước khi áp dụng
- Kiểm tra "Session Status" trong sidebar

### ❌ **DON'T**
- Upload lại cùng file (nó sẽ giữ nguyên)
- F5/Refresh browser (sẽ mất tất cả session)
- Đóng tab trình duyệt

## 🧹 Khi Nào Cần Clear?

### **Tự động clear:**
- Upload file MỚI (khác tên hoặc size)

### **Thủ công clear:**
- Click "🗑️ Xóa Dữ Liệu Hiện Tại" (upload page)
- Click "🗑️ Xóa Tất Cả Cấu Hình" (feature engineering)

## 🔍 Debug Session State

Để xem tất cả session state (development):
```python
import streamlit as st
st.write(st.session_state)
```

## 📊 Ví Dụ Workflow

```
1. Upload data.csv
   → st.session_state.data = data
   → st.session_state.current_file_id = "data.csv_1024"

2. Đi Feature Engineering
   → Cấu hình missing cho column A: Mean
   → st.session_state.missing_config['A'] = {'method': 'Mean', ...}
   → Cấu hình missing cho column B: Median
   → st.session_state.missing_config['B'] = {'method': 'Median', ...}

3. Chuyển sang EDA (xem lại data)
   → Session_state.missing_config VẪN CÒN!
   → Data vẫn ở đó

4. Quay lại Feature Engineering
   → Thấy "📋 2 cấu hình đã lưu"
   → Click "Áp Dụng Tất Cả Cấu Hình"
   → st.session_state.processed_data = data với missing đã fill
   → Config VẪN GIỮ (có thể apply lại hoặc chỉnh sửa)

5. Đi Model Training
   → Train với processed_data
   → st.session_state.model = trained_model
   → Config vẫn còn nếu muốn quay lại chỉnh sửa

6. Muốn làm với dataset khác
   → Về Upload page
   → Click "🗑️ Xóa Dữ Liệu Hiện Tại"
   → TẤT CẢ bị clear
   → Upload file mới
```

## 🚀 Performance Tips

- Session state được lưu trong **memory** của browser session
- Dữ liệu lớn (>100MB) có thể chậm khi pickle/unpickle
- Recommend: Làm việc với sample data trước, sau đó scale lên full dataset

## 🔐 Security Notes

- Session state chỉ tồn tại trong **trình duyệt hiện tại**
- Không share giữa các users
- Không persist sau khi đóng browser
- API keys từ `.env` KHÔNG lưu trong session state

---

**Version**: 1.1.0  
**Last Updated**: 2025-11-13
