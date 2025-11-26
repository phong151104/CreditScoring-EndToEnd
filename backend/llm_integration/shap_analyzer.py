"""
SHAP Analyzer - Phân tích SHAP values bằng LLM
Tích hợp Google Gemini AI để giải thích model và hỗ trợ hỏi đáp
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from .config import LLMConfig


class SHAPAnalyzer:
    """Phân tích SHAP values bằng Google Gemini"""
    
    def __init__(self):
        self.api_key = LLMConfig.GOOGLE_API_KEY
        self.model = LLMConfig.GOOGLE_MODEL
        self.client = None
        self.conversation_history = []
        
    def _init_client(self):
        """Initialize Google Gemini client"""
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Call Google Gemini API"""
        if self.client is None:
            self._init_client()
        
        try:
            import google.generativeai as genai
            generation_config = genai.GenerationConfig(
                max_output_tokens=8000,
                temperature=0.7
            )
            
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            return response.text
                
        except Exception as e:
            return f"Lỗi khi gọi AI: {str(e)}"
    
    def _prepare_shap_context(
        self,
        model_name: str,
        feature_importance: pd.DataFrame,
        shap_values: np.ndarray,
        expected_value: float,
        features: List[str],
        sample_data: pd.DataFrame = None,
        sample_idx: int = None
    ) -> str:
        """Chuẩn bị context về SHAP cho LLM"""
        
        context_parts = []
        
        # Model info
        context_parts.append(f"## Thông Tin Mô Hình")
        context_parts.append(f"- Tên mô hình: {model_name}")
        context_parts.append(f"- Số lượng features: {len(features)}")
        context_parts.append(f"- Số mẫu đã tính SHAP: {len(shap_values)}")
        context_parts.append(f"- Expected value (baseline): {expected_value:.4f}")
        
        # Global feature importance
        context_parts.append(f"\n## Feature Importance (Global)")
        context_parts.append("Top 10 features quan trọng nhất:")
        for i, row in feature_importance.head(10).iterrows():
            context_parts.append(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        # SHAP statistics
        context_parts.append(f"\n## Thống Kê SHAP Values")
        context_parts.append(f"- Mean |SHAP|: {np.abs(shap_values).mean():.4f}")
        context_parts.append(f"- Std SHAP: {np.abs(shap_values).std():.4f}")
        context_parts.append(f"- Max |SHAP|: {np.abs(shap_values).max():.4f}")
        
        # Per-feature SHAP stats
        context_parts.append(f"\n## SHAP Values Theo Feature")
        for i, feat in enumerate(features[:10]):
            feat_shap = shap_values[:, i]
            context_parts.append(f"- {feat}: mean={feat_shap.mean():.4f}, std={feat_shap.std():.4f}, "
                               f"min={feat_shap.min():.4f}, max={feat_shap.max():.4f}")
        
        # Local explanation if provided
        if sample_data is not None and sample_idx is not None:
            sample_shap = shap_values[sample_idx]
            sample_features = sample_data.iloc[sample_idx]
            
            prediction = expected_value + sample_shap.sum()
            prob = 1 / (1 + np.exp(-prediction))
            
            context_parts.append(f"\n## Local Explanation - Mẫu #{sample_idx}")
            context_parts.append(f"- Dự đoán: {prediction:.4f} → Xác suất: {prob:.2%}")
            context_parts.append(f"- Phân loại: {'Rủi ro cao' if prob >= 0.5 else 'Rủi ro thấp'}")
            context_parts.append(f"- Tổng SHAP: {sample_shap.sum():+.4f}")
            
            # Top contributors
            sorted_idx = np.argsort(np.abs(sample_shap))[::-1]
            context_parts.append(f"\nTop đóng góp (theo |SHAP|):")
            for i in sorted_idx[:10]:
                context_parts.append(f"  - {features[i]} = {sample_features.iloc[i]:.2f}: "
                                   f"SHAP = {sample_shap[i]:+.4f}")
            
            # Positive contributors
            pos_idx = np.where(sample_shap > 0)[0]
            if len(pos_idx) > 0:
                pos_sorted = pos_idx[np.argsort(sample_shap[pos_idx])[::-1]][:5]
                context_parts.append(f"\nTác động tích cực (tăng rủi ro):")
                for i in pos_sorted:
                    context_parts.append(f"  - {features[i]} = {sample_features.iloc[i]:.2f}: "
                                       f"SHAP = {sample_shap[i]:+.4f}")
            
            # Negative contributors
            neg_idx = np.where(sample_shap < 0)[0]
            if len(neg_idx) > 0:
                neg_sorted = neg_idx[np.argsort(sample_shap[neg_idx])][:5]
                context_parts.append(f"\nTác động tiêu cực (giảm rủi ro):")
                for i in neg_sorted:
                    context_parts.append(f"  - {features[i]} = {sample_features.iloc[i]:.2f}: "
                                       f"SHAP = {sample_shap[i]:+.4f}")
        
        return "\n".join(context_parts)
    
    def analyze_global(
        self,
        model_name: str,
        feature_importance: pd.DataFrame,
        shap_values: np.ndarray,
        expected_value: float,
        features: List[str]
    ) -> str:
        """
        Phân tích Global SHAP Explanation
        
        Returns:
            Phân tích từ AI (markdown format)
        """
        if self.api_key is None:
            return self._generate_fallback_global(model_name, feature_importance, shap_values, expected_value, features)
        
        context = self._prepare_shap_context(
            model_name, feature_importance, shap_values, expected_value, features
        )
        
        system_prompt = """Bạn là chuyên gia về Machine Learning Explainability, đặc biệt về SHAP (SHapley Additive exPlanations).
Nhiệm vụ của bạn là phân tích và giải thích kết quả SHAP values một cách chuyên nghiệp, dễ hiểu cho người dùng.

Bạn đang làm việc với một hệ thống Credit Scoring - đánh giá rủi ro tín dụng.
- Xác suất cao → Rủi ro cao (khách hàng xấu)
- Xác suất thấp → Rủi ro thấp (khách hàng tốt)
- SHAP dương → Tăng xác suất rủi ro
- SHAP âm → Giảm xác suất rủi ro

Hãy trả lời bằng tiếng Việt, sử dụng markdown format."""

        prompt = f"""Dựa trên dữ liệu SHAP sau đây, hãy phân tích GLOBAL EXPLANATION của mô hình:

{context}

Hãy phân tích theo các phần sau:

## 🌍 Phân Tích Global - Tổng Quan Mô Hình

### 📊 Đặc Trưng Quan Trọng Nhất
- Phân tích top 5 features quan trọng nhất
- Giải thích tại sao chúng quan trọng trong bối cảnh credit scoring

### 💡 Insights Chính
- Mô hình đang học được gì từ dữ liệu?
- Có vấn đề gì về fairness hay bias không?
- Features nào có thể gây overfitting?

### 🎯 Khuyến Nghị
- Những gì cần chú ý khi sử dụng mô hình
- Feature engineering suggestions
- Data quality recommendations

### ⚠️ Lưu Ý
- Các hạn chế của phân tích
- Cần thêm thông tin gì để hiểu rõ hơn"""

        return self._call_llm(prompt, system_prompt)
    
    def analyze_local(
        self,
        model_name: str,
        feature_importance: pd.DataFrame,
        shap_values: np.ndarray,
        expected_value: float,
        features: List[str],
        sample_data: pd.DataFrame,
        sample_idx: int
    ) -> str:
        """
        Phân tích Local SHAP Explanation cho một mẫu cụ thể
        
        Returns:
            Phân tích từ AI (markdown format)
        """
        if self.api_key is None:
            return self._generate_fallback_local(
                model_name, feature_importance, shap_values, 
                expected_value, features, sample_data, sample_idx
            )
        
        context = self._prepare_shap_context(
            model_name, feature_importance, shap_values, expected_value, 
            features, sample_data, sample_idx
        )
        
        system_prompt = """Bạn là chuyên gia về Machine Learning Explainability, đặc biệt về SHAP (SHapley Additive exPlanations).
Nhiệm vụ của bạn là giải thích tại sao mô hình đưa ra dự đoán cụ thể cho một khách hàng.

Bạn đang làm việc với một hệ thống Credit Scoring - đánh giá rủi ro tín dụng.
- Xác suất cao → Rủi ro cao (khách hàng xấu)
- Xác suất thấp → Rủi ro thấp (khách hàng tốt)
- SHAP dương → Tăng xác suất rủi ro
- SHAP âm → Giảm xác suất rủi ro

Hãy trả lời bằng tiếng Việt, sử dụng markdown format, dễ hiểu cho người dùng không chuyên về ML."""

        prompt = f"""Dựa trên dữ liệu SHAP sau đây, hãy giải thích LOCAL EXPLANATION cho mẫu cụ thể:

{context}

Hãy phân tích theo các phần sau:

## 🎯 Phân Tích Mẫu #{sample_idx}

### 📋 Kết Quả Dự Đoán
- Tóm tắt kết quả dự đoán
- Giải thích ý nghĩa của xác suất

### 🔍 Các Yếu Tố Chính
- Phân tích các yếu tố làm TĂNG rủi ro
- Phân tích các yếu tố làm GIẢM rủi ro
- So sánh với baseline (expected value)

### 💭 Giải Thích Tổng Hợp
- Tại sao mô hình đưa ra quyết định này?
- Kể một "câu chuyện" về khách hàng này dựa trên dữ liệu

### 🎯 Đề Xuất
- Nếu khách hàng muốn cải thiện, nên làm gì?
- Những yếu tố nào có thể thay đổi được?

### ⚠️ Lưu Ý
- Độ tin cậy của dự đoán
- Các yếu tố bất thường (nếu có)"""

        return self._call_llm(prompt, system_prompt)
    
    def chat(
        self,
        user_question: str,
        model_name: str,
        feature_importance: pd.DataFrame,
        shap_values: np.ndarray,
        expected_value: float,
        features: List[str],
        sample_data: pd.DataFrame = None,
        conversation_history: List[Dict] = None
    ) -> str:
        """
        Chat với AI về mô hình và SHAP values
        
        Args:
            user_question: Câu hỏi từ người dùng
            ... các context khác
            conversation_history: Lịch sử hội thoại
            
        Returns:
            Câu trả lời từ AI
        """
        if self.api_key is None:
            return self._generate_fallback_chat(user_question, model_name, feature_importance)
        
        context = self._prepare_shap_context(
            model_name, feature_importance, shap_values, expected_value, features
        )
        
        system_prompt = f"""Bạn là AI Assistant chuyên về Machine Learning Explainability, đặc biệt về SHAP values.
Bạn đang hỗ trợ người dùng hiểu về mô hình Credit Scoring.

CONTEXT VỀ MÔ HÌNH:
{context}

QUY TẮC:
1. Trả lời bằng tiếng Việt
2. Dựa trên dữ liệu SHAP thực tế được cung cấp
3. Giải thích rõ ràng, dễ hiểu cho người không chuyên
4. Nếu không chắc chắn, hãy nói rõ
5. Sử dụng markdown format cho câu trả lời"""

        # Build conversation context
        conv_context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = "Người dùng" if msg.get("role") == "user" else "AI"
                conv_context += f"\n{role}: {msg.get('content', '')}\n"
        
        prompt = f"""{"Lịch sử hội thoại:" + conv_context if conv_context else ""}

Câu hỏi mới: {user_question}

Hãy trả lời câu hỏi dựa trên context về SHAP và mô hình đã cung cấp."""

        return self._call_llm(prompt, system_prompt)
    
    def _generate_fallback_global(
        self, model_name: str, feature_importance: pd.DataFrame,
        shap_values: np.ndarray, expected_value: float, features: List[str]
    ) -> str:
        """Fallback khi không có API key"""
        top_features = feature_importance.head(5)
        total_importance = top_features['Importance'].sum()
        
        response = f"""## 🌍 Phân Tích Global - Tổng Quan Mô Hình {model_name}

### 📊 Đặc Trưng Quan Trọng Nhất

"""
        for i, row in top_features.iterrows():
            pct = row['Importance'] / total_importance * 100
            response += f"**{i+1}. {row['Feature']}** (Impact: {row['Importance']:.4f})\n"
            response += f"   - Chiếm {pct:.1f}% trong top 5 features\n\n"
        
        response += f"""### 💡 Insights Chính

- Mô hình **{model_name}** sử dụng {len(features)} đặc trưng để dự đoán
- Expected value (baseline): {expected_value:.4f}
- Top feature **{feature_importance.iloc[0]['Feature']}** có ảnh hưởng lớn nhất
- Mean |SHAP|: {np.abs(shap_values).mean():.4f}

### 🎯 Khuyến Nghị

1. Đảm bảo chất lượng dữ liệu cho top features
2. Theo dõi sự thay đổi feature importance theo thời gian
3. Xem xét thêm feature engineering cho các biến quan trọng

### ⚠️ Lưu Ý

*Đây là phân tích tự động. Để có phân tích chi tiết hơn từ Google Gemini AI, vui lòng cấu hình GOOGLE_API_KEY trong file .env*
"""
        return response
    
    def _generate_fallback_local(
        self, model_name: str, feature_importance: pd.DataFrame,
        shap_values: np.ndarray, expected_value: float, features: List[str],
        sample_data: pd.DataFrame, sample_idx: int
    ) -> str:
        """Fallback cho local explanation khi không có API key"""
        sample_shap = shap_values[sample_idx]
        sample_features = sample_data.iloc[sample_idx]
        
        prediction = expected_value + sample_shap.sum()
        prob = 1 / (1 + np.exp(-prediction))
        
        # Top positive
        pos_idx = np.where(sample_shap > 0)[0]
        neg_idx = np.where(sample_shap < 0)[0]
        
        response = f"""## 🎯 Phân Tích Mẫu #{sample_idx}

### 📋 Kết Quả Dự Đoán

- **Xác suất rủi ro**: {prob:.1%}
- **Phân loại**: {"⚠️ Rủi ro cao" if prob >= 0.5 else "✅ Rủi ro thấp"}
- **Base value**: {expected_value:.4f}
- **Tổng SHAP**: {sample_shap.sum():+.4f}

### 🔍 Các Yếu Tố Chính

**Tác động tích cực (TĂNG rủi ro):**
"""
        if len(pos_idx) > 0:
            pos_sorted = pos_idx[np.argsort(sample_shap[pos_idx])[::-1]][:3]
            for i in pos_sorted:
                response += f"- **{features[i]}** = {sample_features.iloc[i]:.2f}: SHAP = {sample_shap[i]:+.4f}\n"
        else:
            response += "- Không có yếu tố nào làm tăng rủi ro\n"
        
        response += "\n**Tác động tiêu cực (GIẢM rủi ro):**\n"
        if len(neg_idx) > 0:
            neg_sorted = neg_idx[np.argsort(sample_shap[neg_idx])][:3]
            for i in neg_sorted:
                response += f"- **{features[i]}** = {sample_features.iloc[i]:.2f}: SHAP = {sample_shap[i]:+.4f}\n"
        else:
            response += "- Không có yếu tố nào làm giảm rủi ro\n"
        
        response += f"""
### 💭 Giải Thích Tổng Hợp

Mẫu này có xác suất rủi ro {prob:.1%}. 
Yếu tố quyết định chính là **{features[np.argmax(np.abs(sample_shap))]}**.

### ⚠️ Lưu Ý

*Đây là phân tích tự động. Để có phân tích chi tiết hơn từ Google Gemini AI, vui lòng cấu hình GOOGLE_API_KEY trong file .env*
"""
        return response
    
    def _generate_fallback_chat(
        self, question: str, model_name: str, feature_importance: pd.DataFrame
    ) -> str:
        """Fallback cho chat khi không có API key"""
        top_feat = feature_importance.iloc[0]['Feature']
        top_imp = feature_importance.iloc[0]['Importance']
        
        return f"""**Câu hỏi:** {question}

**Trả lời:**

Dựa trên phân tích SHAP của mô hình **{model_name}**:

- Yếu tố quan trọng nhất là **{top_feat}** với mean |SHAP| = {top_imp:.4f}
- Tổng cộng có {len(feature_importance)} features được sử dụng

**⚠️ Lưu ý:** Đây là câu trả lời tự động. Để có phân tích chi tiết hơn từ AI thật, vui lòng cấu hình GOOGLE_API_KEY trong file `.env`.

Xem hướng dẫn tại file `env.example`."""


def create_shap_analyzer() -> SHAPAnalyzer:
    """Factory function để tạo SHAPAnalyzer"""
    return SHAPAnalyzer()
