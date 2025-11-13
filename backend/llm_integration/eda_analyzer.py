"""
EDA Analyzer - Thu thập và phân tích dữ liệu EDA bằng LLM
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json


class EDADataCollector:
    """Thu thập toàn bộ thông tin từ quá trình EDA"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.summary = {}
        
    def collect_basic_info(self) -> Dict[str, Any]:
        """Thu thập thông tin cơ bản về dataset"""
        return {
            "n_rows": len(self.data),
            "n_columns": len(self.data.columns),
            "columns": self.data.columns.tolist(),
            "dtypes": self.data.dtypes.astype(str).to_dict(),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / 1024**2
        }
    
    def collect_missing_data(self) -> Dict[str, Any]:
        """Phân tích dữ liệu thiếu"""
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data) * 100).round(2)
        
        return {
            "total_missing": int(missing.sum()),
            "missing_by_column": {
                col: {
                    "count": int(missing[col]),
                    "percentage": float(missing_pct[col])
                }
                for col in self.data.columns if missing[col] > 0
            },
            "columns_with_missing": missing[missing > 0].index.tolist(),
            "complete_rows": int((~self.data.isnull().any(axis=1)).sum())
        }
    
    def collect_numeric_stats(self) -> Dict[str, Any]:
        """Thống kê chi tiết cho biến số"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        stats = {}
        
        for col in numeric_cols:
            col_data = self.data[col].dropna()
            
            if len(col_data) == 0:
                continue
                
            # Calculate statistics
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # Skewness and kurtosis
            try:
                from scipy import stats as scipy_stats
                skewness = float(scipy_stats.skew(col_data))
                kurtosis = float(scipy_stats.kurtosis(col_data))
            except:
                skewness = None
                kurtosis = None
            
            stats[col] = {
                "count": int(col_data.count()),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "outliers_count": len(outliers),
                "outliers_pct": round(len(outliers) / len(col_data) * 100, 2),
                "skewness": skewness,
                "kurtosis": kurtosis,
                "cv": round(col_data.std() / col_data.mean() * 100, 2) if col_data.mean() != 0 else None,
                "zeros_count": int((col_data == 0).sum()),
                "zeros_pct": round((col_data == 0).sum() / len(col_data) * 100, 2)
            }
        
        return stats
    
    def collect_categorical_stats(self) -> Dict[str, Any]:
        """Thống kê chi tiết cho biến phân loại"""
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
        stats = {}
        
        for col in cat_cols:
            col_data = self.data[col].dropna()
            
            if len(col_data) == 0:
                continue
            
            value_counts = col_data.value_counts()
            
            stats[col] = {
                "unique_count": int(col_data.nunique()),
                "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "most_common_pct": round(value_counts.iloc[0] / len(col_data) * 100, 2) if len(value_counts) > 0 else 0,
                "top_5_values": value_counts.head(5).to_dict(),
                "is_high_cardinality": col_data.nunique() > len(col_data) * 0.5,
                "entropy": float(-sum((value_counts / len(col_data)) * np.log2(value_counts / len(col_data))))
            }
        
        return stats
    
    def collect_correlations(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Phân tích tương quan giữa các biến"""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return {"message": "Không đủ biến số để tính tương quan"}
        
        corr_matrix = numeric_data.corr()
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": round(float(corr_val), 3),
                        "type": "positive" if corr_val > 0 else "negative",
                        "strength": "strong" if abs(corr_val) >= 0.7 else "moderate"
                    })
        
        # Sort by absolute correlation
        high_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            "correlation_matrix_shape": corr_matrix.shape,
            "high_correlations": high_corr,
            "avg_correlation": round(float(corr_matrix.abs().mean().mean()), 3),
            "max_correlation": round(float(corr_matrix.abs().max().max()), 3) if len(corr_matrix) > 1 else 0
        }
    
    def collect_data_quality_issues(self) -> Dict[str, List[str]]:
        """Phát hiện các vấn đề về chất lượng dữ liệu"""
        issues = {
            "high_missing": [],
            "high_cardinality": [],
            "potential_id_columns": [],
            "constant_columns": [],
            "high_outliers": [],
            "highly_skewed": [],
            "duplicate_rows": []
        }
        
        # High missing values (>30%)
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100)
        issues["high_missing"] = missing_pct[missing_pct > 30].index.tolist()
        
        # High cardinality categorical
        for col in self.data.select_dtypes(include=['object', 'category']).columns:
            if self.data[col].nunique() > len(self.data) * 0.5:
                issues["high_cardinality"].append(col)
        
        # Potential ID columns (all unique numeric)
        for col in self.data.select_dtypes(include=[np.number]).columns:
            if self.data[col].nunique() == len(self.data):
                issues["potential_id_columns"].append(col)
        
        # Constant columns (only 1 unique value)
        for col in self.data.columns:
            if self.data[col].nunique() == 1:
                issues["constant_columns"].append(col)
        
        # High outliers (>10% outliers)
        for col in self.data.select_dtypes(include=[np.number]).columns:
            col_data = self.data[col].dropna()
            if len(col_data) > 0:
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]
                if len(outliers) / len(col_data) > 0.1:
                    issues["high_outliers"].append(col)
        
        # Highly skewed (|skewness| > 2)
        try:
            from scipy import stats as scipy_stats
            for col in self.data.select_dtypes(include=[np.number]).columns:
                col_data = self.data[col].dropna()
                if len(col_data) > 0:
                    skew = scipy_stats.skew(col_data)
                    if abs(skew) > 2:
                        issues["highly_skewed"].append(col)
        except:
            pass
        
        # Duplicate rows
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            issues["duplicate_rows"] = [f"{duplicates} duplicate rows found ({round(duplicates/len(self.data)*100, 2)}%)"]
        
        return issues
    
    def generate_full_summary(self) -> Dict[str, Any]:
        """Tạo báo cáo tổng hợp toàn bộ EDA"""
        return {
            "basic_info": self.collect_basic_info(),
            "missing_data": self.collect_missing_data(),
            "numeric_stats": self.collect_numeric_stats(),
            "categorical_stats": self.collect_categorical_stats(),
            "correlations": self.collect_correlations(),
            "data_quality_issues": self.collect_data_quality_issues()
        }
    
    def to_text_summary(self, compact: bool = False) -> str:
        """
        Chuyển đổi summary thành văn bản dễ đọc cho LLM
        
        Args:
            compact: If True, return compact version (limit details for large datasets)
        """
        summary = self.generate_full_summary()
        
        text_parts = []
        
        # Basic Info
        text_parts.append("=" * 80)
        text_parts.append("THÔNG TIN CƠ BẢN VỀ DATASET")
        text_parts.append("=" * 80)
        basic = summary['basic_info']
        text_parts.append(f"Số dòng: {basic['n_rows']:,}")
        text_parts.append(f"Số cột: {basic['n_columns']}")
        text_parts.append(f"Dung lượng: {basic['memory_usage_mb']:.2f} MB")
        
        # Compact mode: only list column names count
        if compact and len(basic['columns']) > 15:
            text_parts.append(f"\nCác cột ({len(basic['columns'])} cột): {', '.join(basic['columns'][:10])}... (và {len(basic['columns'])-10} cột khác)")
        else:
            text_parts.append(f"\nCác cột: {', '.join(basic['columns'])}")
        
        # Missing Data
        text_parts.append("\n" + "=" * 80)
        text_parts.append("PHÂN TÍCH DỮ LIỆU THIẾU")
        text_parts.append("=" * 80)
        missing = summary['missing_data']
        text_parts.append(f"Tổng giá trị thiếu: {missing['total_missing']:,}")
        text_parts.append(f"Số dòng hoàn chỉnh: {missing['complete_rows']:,}")
        if missing['missing_by_column']:
            text_parts.append("\nCác cột có dữ liệu thiếu:")
            for col, info in missing['missing_by_column'].items():
                text_parts.append(f"  - {col}: {info['count']} ({info['percentage']:.2f}%)")
        
        # Numeric Stats
        text_parts.append("\n" + "=" * 80)
        text_parts.append("THỐNG KÊ BIẾN SỐ")
        text_parts.append("=" * 80)
        for col, stats in summary['numeric_stats'].items():
            text_parts.append(f"\n{col}:")
            text_parts.append(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}, Std: {stats['std']:.2f}")
            text_parts.append(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            text_parts.append(f"  Outliers: {stats['outliers_count']} ({stats['outliers_pct']:.2f}%)")
            if stats['skewness'] is not None:
                text_parts.append(f"  Skewness: {stats['skewness']:.2f}, Kurtosis: {stats['kurtosis']:.2f}")
            if stats['cv'] is not None:
                text_parts.append(f"  Coefficient of Variation: {stats['cv']:.2f}%")
        
        # Categorical Stats
        text_parts.append("\n" + "=" * 80)
        text_parts.append("THỐNG KÊ BIẾN PHÂN LOẠI")
        text_parts.append("=" * 80)
        for col, stats in summary['categorical_stats'].items():
            text_parts.append(f"\n{col}:")
            text_parts.append(f"  Số giá trị khác nhau: {stats['unique_count']}")
            text_parts.append(f"  Giá trị phổ biến nhất: {stats['most_common']} ({stats['most_common_pct']:.2f}%)")
            text_parts.append(f"  High cardinality: {'Có' if stats['is_high_cardinality'] else 'Không'}")
        
        # Correlations
        text_parts.append("\n" + "=" * 80)
        text_parts.append("PHÂN TÍCH TƯƠNG QUAN")
        text_parts.append("=" * 80)
        corr = summary['correlations']
        if 'high_correlations' in corr and corr['high_correlations']:
            text_parts.append(f"Số cặp biến có tương quan cao (≥0.5): {len(corr['high_correlations'])}")
            text_parts.append("\nTop correlations:")
            for item in corr['high_correlations'][:10]:
                text_parts.append(f"  - {item['var1']} ↔ {item['var2']}: {item['correlation']:.3f} ({item['strength']}, {item['type']})")
        
        # Data Quality Issues
        text_parts.append("\n" + "=" * 80)
        text_parts.append("VẤN ĐỀ CHẤT LƯỢNG DỮ LIỆU")
        text_parts.append("=" * 80)
        issues = summary['data_quality_issues']
        for issue_type, issue_list in issues.items():
            if issue_list:
                text_parts.append(f"\n{issue_type.replace('_', ' ').title()}:")
                for item in issue_list:
                    text_parts.append(f"  - {item}")
        
        return "\n".join(text_parts)


class LLMEDAAnalyzer:
    """Sử dụng LLM để phân tích kết quả EDA"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash", provider: str = "google"):
        """
        Initialize LLM analyzer
        
        Args:
            api_key: API key for LLM service (OpenAI, Anthropic, Google)
            model: Model name to use
            provider: LLM provider ('openai', 'anthropic', 'google')
        """
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.client = None
        
    def _init_client(self):
        """Initialize LLM client based on provider"""
        if self.api_key is None:
            raise ValueError("API key is required. Set it in environment or pass to constructor.")
        
        if self.provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("Google Generative AI library not installed. Run: pip install google-generativeai")
        
        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic library not installed. Run: pip install anthropic")
    
    def create_analysis_prompt(self, eda_summary: str) -> str:
        """Tạo prompt cho LLM để phân tích EDA"""
        prompt = f"""Bạn là một Data Scientist chuyên nghiệp với nhiều năm kinh nghiệm trong phân tích dữ liệu và xây dựng mô hình Credit Scoring.

Dựa trên kết quả EDA (Exploratory Data Analysis) dưới đây, hãy cung cấp một phân tích chi tiết và đề xuất các bước tiền xử lý cần thiết.

{eda_summary}

Hãy phân tích và trả lời theo cấu trúc sau:

## 1. ĐÁNH GIÁ TỔNG QUAN
- Chất lượng tổng thể của dataset
- Các điểm mạnh và điểm yếu
- Mức độ sẵn sàng cho modeling

## 2. PHÂN TÍCH CHI TIẾT

### 2.1 Dữ Liệu Thiếu (Missing Data)
- Đánh giá mức độ nghiêm trọng
- Nguyên nhân có thể
- Đề xuất phương pháp xử lý (imputation, deletion, etc.)

### 2.2 Biến Số (Numeric Variables)
- Phân phối của các biến (normal, skewed, etc.)
- Outliers và cách xử lý
- Các biến cần transformation (log, sqrt, standardization, etc.)

### 2.3 Biến Phân Loại (Categorical Variables)
- Vấn đề về cardinality
- Encoding strategy (one-hot, label, target encoding)
- Xử lý rare categories

### 2.4 Tương Quan (Correlations)
- Multicollinearity issues
- Feature selection recommendations
- Potential feature engineering opportunities

## 3. VẤN ĐỀ CẦN ƯU TIÊN XỬ LÝ
Liệt kê các vấn đề theo thứ tự ưu tiên:
1. [Vấn đề 1]
2. [Vấn đề 2]
...

## 4. ROADMAP TIỀN XỬ LÝ
Đề xuất các bước cụ thể cần thực hiện:
- Bước 1: ...
- Bước 2: ...
...

## 5. KẾT LUẬN
- Tóm tắt đánh giá
- Dự đoán khả năng xây dựng mô hình tốt
- Các lưu ý đặc biệt

Hãy trả lời bằng tiếng Việt, chuyên nghiệp nhưng dễ hiểu."""

        return prompt
    
    def analyze(self, data: pd.DataFrame, use_cached: bool = True) -> str:
        """
        Phân tích dữ liệu EDA bằng LLM
        
        Args:
            data: DataFrame cần phân tích
            use_cached: Sử dụng kết quả cached nếu có
            
        Returns:
            Phân tích chi tiết từ LLM (markdown format)
        """
        # Collect EDA data
        collector = EDADataCollector(data)
        
        # Use compact mode for large datasets (>20 columns)
        compact = len(data.columns) > 20
        eda_summary = collector.to_text_summary(compact=compact)
        
        # If no API key, return template analysis
        if self.api_key is None:
            return self._generate_template_analysis(collector.generate_full_summary())
        
        # Initialize client
        if self.client is None:
            self._init_client()
        
        # Create prompt
        prompt = self.create_analysis_prompt(eda_summary)
        
        # Call LLM based on provider
        try:
            if self.provider == "google":
                # Google Gemini with timeout
                import google.generativeai as genai
                
                # Configure timeout and safety settings
                generation_config = genai.GenerationConfig(
                    max_output_tokens=8000,
                    temperature=0.7,
                )
                
                # Add timeout wrapper
                import time
                start_time = time.time()
                timeout = 200  # 200 seconds timeout for complex analysis
                
                response = self.client.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={'timeout': timeout}
                )
                
                elapsed = time.time() - start_time
                print(f"✓ LLM response received in {elapsed:.1f}s")
                
                return response.text
            
            elif self.provider == "openai":
                # OpenAI GPT
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Bạn là một Data Scientist chuyên nghiệp."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=8000
                )
                return response.choices[0].message.content
            
            elif self.provider == "anthropic":
                # Anthropic Claude
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8000,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
            else:
                return f"❌ **Provider không được hỗ trợ**: {self.provider}"
        
        except Exception as e:
            return f"❌ **Lỗi khi gọi LLM**: {str(e)}\n\nVui lòng kiểm tra API key và thử lại."
    
    def _generate_template_analysis(self, summary: Dict[str, Any]) -> str:
        """Tạo phân tích mẫu khi không có API key"""
        basic = summary['basic_info']
        missing = summary['missing_data']
        issues = summary['data_quality_issues']
        
        template = f"""## 🔍 PHÂN TÍCH TỰ ĐỘNG (TEMPLATE MODE)

> ⚠️ **Lưu ý**: Đây là phân tích tự động cơ bản. Để có phân tích chi tiết từ AI, vui lòng cấu hình API key.

### 1. ĐÁNH GIÁ TỔNG QUAN

Dataset có **{basic['n_rows']:,} dòng** và **{basic['n_columns']} cột**.

**Chất lượng dữ liệu**: {'⚠️ Cần cải thiện' if missing['total_missing'] > 0 else '✅ Tốt'}

### 2. VẤN ĐỀ PHÁT HIỆN

"""
        
        # Missing data issues
        if missing['total_missing'] > 0:
            template += f"\n#### 📉 Dữ Liệu Thiếu\n"
            template += f"- Tổng: **{missing['total_missing']:,}** giá trị thiếu\n"
            template += f"- Số cột bị ảnh hưởng: **{len(missing['columns_with_missing'])}**\n"
        
        # Data quality issues
        critical_issues = []
        if issues['high_missing']:
            critical_issues.append(f"**{len(issues['high_missing'])} cột** có >30% dữ liệu thiếu")
        if issues['constant_columns']:
            critical_issues.append(f"**{len(issues['constant_columns'])} cột** có giá trị không đổi")
        if issues['high_outliers']:
            critical_issues.append(f"**{len(issues['high_outliers'])} cột** có nhiều outliers (>10%)")
        if issues['highly_skewed']:
            critical_issues.append(f"**{len(issues['highly_skewed'])} cột** có phân phối lệch mạnh")
        
        if critical_issues:
            template += "\n#### ⚠️ Vấn Đề Cần Xử Lý\n"
            for issue in critical_issues:
                template += f"- {issue}\n"
        
        # Recommendations
        template += "\n### 3. ĐỀ XUẤT TIỀN XỬ LÝ\n\n"
        
        if missing['total_missing'] > 0:
            template += "**Xử lý dữ liệu thiếu:**\n"
            template += "- Xem xét imputation (mean/median cho số, mode cho categorical)\n"
            template += "- Hoặc loại bỏ các dòng/cột có quá nhiều missing\n\n"
        
        if issues['high_outliers']:
            template += "**Xử lý outliers:**\n"
            template += "- Cân nhắc winsorization hoặc transformation (log, sqrt)\n"
            template += "- Kiểm tra xem có phải outliers hợp lệ không\n\n"
        
        if issues['highly_skewed']:
            template += "**Xử lý phân phối lệch:**\n"
            template += "- Áp dụng log/sqrt transformation\n"
            template += "- Xem xét standardization sau transformation\n\n"
        
        template += "\n### 4. KẾT LUẬN\n\n"
        template += "Dataset cần **tiền xử lý** trước khi training model. "
        template += "Hãy thực hiện các bước đề xuất ở phần Feature Engineering.\n\n"
        template += "---\n"
        template += "*💡 Để có phân tích chi tiết hơn từ AI, hãy cấu hình OpenAI API key trong file `.env`*"
        
        return template


# Utility functions
def analyze_eda_with_llm(data: pd.DataFrame, api_key: Optional[str] = None, provider: str = "google") -> str:
    """
    Quick function để phân tích EDA
    
    Args:
        data: DataFrame to analyze
        api_key: API key (if None, will use from config)
        provider: LLM provider ('google', 'openai', 'anthropic')
    
    Usage:
        # Google Gemini
        analysis = analyze_eda_with_llm(df, api_key="...", provider="google")
        
        # OpenAI GPT
        analysis = analyze_eda_with_llm(df, api_key="sk-...", provider="openai")
    """
    from .config import LLMConfig
    
    # Get API key and model from config if not provided
    if api_key is None:
        api_key = LLMConfig.get_api_key(provider)
    
    model = LLMConfig.get_model(provider)
    
    analyzer = LLMEDAAnalyzer(api_key=api_key, model=model, provider=provider)
    return analyzer.analyze(data)


def get_eda_summary(data: pd.DataFrame, format: str = "text") -> str:
    """
    Lấy summary của EDA
    
    Args:
        data: DataFrame
        format: "text" or "json"
    """
    collector = EDADataCollector(data)
    
    if format == "json":
        return json.dumps(collector.generate_full_summary(), indent=2, ensure_ascii=False)
    else:
        return collector.to_text_summary()
