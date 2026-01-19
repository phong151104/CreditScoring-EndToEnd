# 🔐 Phân Quyền Hệ Thống Credit Scoring

Tài liệu này mô tả chi tiết các vai trò (Roles) và quyền hạn tương ứng trong hệ thống.

## 1. 👨‍💼 Admin (Quản trị viên)
**Quyền hạn cao nhất**, quản lý toàn bộ hệ thống.

*   **Truy cập:** Tất cả các trang.
*   **Chức năng:**
    *   ✅ **Quản lý User:** Thêm, sửa, xóa người dùng, cấp lại mật khẩu.
    *   ✅ **Upload Data:** Tải lên dữ liệu train/test, xóa dữ liệu cũ.
    *   ✅ **Feature Engineering:** Thực hiện xử lý dữ liệu, feature selection, binning, WOE/IV.
    *   ✅ **Model Training:** Huấn luyện mô hình, chạy Cross-validation, Tuning tham số.
    *   ✅ **Model Explanation:** Sử dụng SHAP để giải thích mô hình.
    *   ✅ **Prediction:** Chấm điểm tín dụng cho khách hàng mới.
    *   ✅ **Cài đặt hệ thống:** Cấu hình ngưỡng duyệt vay, công thức tính điểm (nếu có).

---

## 2. 👷 Model Builder (Xây dựng mô hình)
Chuyên gia dữ liệu, tập trung vào việc xây dựng và tối ưu mô hình, không can thiệp vào quản trị hệ thống.

*   **Truy cập:** Hầu hết các trang (Trừ trang *Admin Settings*).
*   **Chức năng:**
    *   ✅ **Upload Data:** Tải lên và quản lý dữ liệu.
    *   ✅ **Feature Engineering:** Full quyền xử lý dữ liệu.
    *   ✅ **Model Training:** Full quyền huấn luyện và tinh chỉnh mô hình.
    *   ✅ **Model Explanation:** Full quyền phân tích SHAP.
    *   ✅ **Prediction:** Test chấm điểm.
    *   ❌ **Quản trị:** KHÔNG thể quản lý User hay thay đổi cấu hình hệ thống cấp cao.

---

## 3. 👨‍🔬 Validator (Kiểm định viên)
Người đánh giá độc lập, có quyền xem chi tiết mọi thứ để thẩm định nhưng không được phép thay đổi dữ liệu hay mô hình.

*   **Truy cập:** Dashboard, Data Analysis, Feature Engineering, Training, Explanation, Prediction.
*   **Chế độ:** **👀 View-only** (Chỉ xem).
*   **Chức năng:**
    *   ✅ **Xem:** Xem dữ liệu, xem biểu đồ EDA, xem cấu hình Feature Engineering đã thực hiện.
    *   ✅ **Review Model:** Xem kết quả Training, metrics, biểu đồ so sánh.
    *   ✅ **SHAP:** Xem giải thích mô hình, sử dụng tính năng "Tạo Phân Tích AI".
    *   ✅ **Prediction:** Xem trang dự đoán (nhưng không thực hiện dự đoán - *View only*).
    *   ❌ **Thao tác:** KHÔNG thể upload, training, tuning, hay thay đổi bất kỳ cấu hình nào. Các nút bấm chức năng quan trọng đều bị vô hiệu hóa.

---

## 4. 👨‍💻 Scorer (Người chấm điểm / Cán bộ tín dụng)
Người dùng cuối, chỉ sử dụng mô hình đã *deploy* để chấm điểm khách hàng.

*   **Truy cập:** Chỉ trang **🎯 Prediction & Advisory**.
*   **Chức năng:**
    *   ✅ **Prediction:** Nhập thông tin khách hàng -> Nhận điểm tín dụng và kết quả (Duyệt/Từ chối).
    *   ✅ **Advisory:** Xem gợi ý cải thiện điểm số.
    *   ❌ **Hệ thống:** KHÔNG thấy các trang kỹ thuật (Data, Training, Feature...).

---

## 🔑 Bảng Tóm Tắt Quyền Truy Cập Trang

| Trang Chức Năng | Admin | Model Builder | Validator | Scorer |
| :--- | :---: | :---: | :---: | :---: |
| 🏠 Dashboard | ✅ | ✅ | ✅ | ❌ |
| 📊 Data Upload & Analysis | ✅ | ✅ | ✅ (View) | ❌ |
| ⚙️ Feature Engineering | ✅ | ✅ | ✅ (View) | ❌ |
| 🧠 Model Training | ✅ | ✅ | ✅ (View) | ❌ |
| 💡 Model Explanation | ✅ | ✅ | ✅ (View) | ❌ |
| 🎯 Prediction & Advisory | ✅ | ✅ | ✅ (View) | ✅ |
| ⚡ Admin Settings | ✅ | ❌ | ❌ | ❌ |
