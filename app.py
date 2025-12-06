import streamlit as st
import re
import os
from datetime import datetime
from vietnamese_sentiment import VietnameseSentimentAnalyzer
from database import (
    get_timestamp, insert_sentiment_analysis, 
    init_database, get_sentiment_analysis
)

# Khởi tạo database
init_database()


# Import các thư viện ML 
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import numpy as np
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False
    st.warning("⚠️ Các thư viện ML chưa được cài đặt. Chạy: pip install -r requirements.txt")

# Cấu hình trang
st.set_page_config(
    page_title="Phân Loại Cảm Xúc Văn Bản",
    page_icon="😊",
    layout="wide"
)

# CSS tùy chỉnh để làm đẹp UI
st.markdown("""
    <style>
    .main-header {
        font-size: 9.5rem;
        font-weight: 900;
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1.5rem;
        padding: 1.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: 2px;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.4rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .emotion-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .positive { background-color: #d4edda; color: #155724; }
    .negative { background-color: #f8d7da; color: #721c24; }
    .neutral { background-color: #d1ecf1; color: #0c5460; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Trợ lý phân loại cảm xúc văn bản tiếng Việt</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Nhập văn bản của bạn để phân loại cảm xúc</p>', unsafe_allow_html=True)

# Tabs cho Phân loại và Lịch sử
tab1, tab2 = st.tabs(["🔍 Phân loại", "📜 Lịch sử"])

# Khởi tạo session state cho lịch sử và analyzer
if 'analyzer_loaded' not in st.session_state:
    st.session_state.analyzer_loaded = False
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

# Sidebar với thông tin
with st.sidebar:
    st.header("ℹ️ Thông tin")
    st.info("""
    Ứng dụng này giúp bạn phân loại cảm xúc trong văn bản.
    
    **Các loại cảm xúc:**
    - 😊 Tích cực (Positive)
    - 😢 Tiêu cực (Negative)
    - 😐 Trung tính (Neutral)
    """)

# Hàm load Vietnamese Sentiment Analyzer với caching
@st.cache_resource
def load_sentiment_analyzer():
    """
    Load Vietnamese Sentiment Analyzer
    """
    if not ML_LIBRARIES_AVAILABLE:
        return None
    try:
        analyzer = VietnameseSentimentAnalyzer()
        return analyzer
    except Exception as e:
        st.error(f"Lỗi khi load analyzer: {str(e)}")
        return None

def validate_text(text):
    """
    Validate văn bản đầu vào
    
    Args:
        text: Văn bản cần kiểm tra
        
    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    if not text or len(text.strip()) == 0:
        return False, "⚠️ Văn bản không được để trống!"
    
    # Loại bỏ khoảng trắng để kiểm tra độ dài thực tế
    text_no_spaces = text.replace(" ", "").replace("\n", "").replace("\t", "")
    
    # Kiểm tra độ dài tối thiểu (ít nhất 5 ký tự)
    if len(text_no_spaces) < 5:
        return False, "⚠️ Văn bản phải có ít nhất 5 ký tự (không tính khoảng trắng)!"
    
    # Kiểm tra xem có phải là văn bản vô nghĩa không
    # - Chỉ chứa ký tự lặp lại (ví dụ: "aaaaa", "11111")
    if len(set(text_no_spaces.lower())) <= 2 and len(text_no_spaces) > 5:
        return False, "⚠️ Văn bản không hợp lệ! Vui lòng nhập văn bản có ý nghĩa."
    
    # Kiểm tra xem có chứa ít nhất một ký tự chữ cái (a-z, A-Z, hoặc tiếng Việt)
    import re
    has_letters = bool(re.search(r'[a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]', text))
    if not has_letters:
        return False, "⚠️ Văn bản phải chứa ít nhất một ký tự chữ cái!"
    
    # Kiểm tra xem có quá nhiều ký tự đặc biệt không (ví dụ: "!@#$%^&*()")
    special_chars = len(re.findall(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]', text))
    if special_chars > len(text) * 0.5:  # Nếu hơn 50% là ký tự đặc biệt
        return False, "⚠️ Văn bản chứa quá nhiều ký tự đặc biệt!"
    
    return True, ""

# Gọi hàm phân loại cảm xúc vào function classify_emotion
def classify_emotion(text):
    """
    Phân loại cảm xúc sử dụng Vietnamese Sentiment Analyzer
    """
    if not text or len(text.strip()) == 0:
        return None
    
    # Load analyzer nếu chưa có
    if st.session_state.analyzer is None:
        st.session_state.analyzer = load_sentiment_analyzer()
    
    if st.session_state.analyzer is None:
        return None
    
    return st.session_state.analyzer.analyze_sentiment(text)

# Map sentiment label -> tiếng việt
def map_sentiment_label(sentiment_label):
    """
    Map sentiment label từ model sang tiếng Việt
    """
    sentiment_label = sentiment_label.upper()
    if "POS" in sentiment_label or "POSITIVE" in sentiment_label:
        return "Tích cực", "😊"
    elif "NEG" in sentiment_label or "NEGATIVE" in sentiment_label:
        return "Tiêu cực", "😢"
    else:
        return "Trung tính", "😐"

# Tạo dictionary điểm số cho các cảm xúc
def create_scores_dict(sentiment_label, confidence, all_scores=None):
    """
    Tạo dictionary điểm số cho các cảm xúc
    
    Args:
        sentiment_label: Label được dự đoán (NEG/POS/NEU)
        confidence: Confidence score của label được dự đoán
        all_scores: Dict chứa scores thực từ model cho tất cả labels (nếu có)
    
    Returns:
        dict: Dictionary với scores cho "Tích cực", "Tiêu cực", "Trung tính"
    """
    # Map labels từ model sang tiếng Việt
    label_mapping = {
        "POS": "Tích cực",
        "POSITIVE": "Tích cực",
        "NEG": "Tiêu cực", 
        "NEGATIVE": "Tiêu cực",
        "NEU": "Trung tính",
        "NEUTRAL": "Trung tính"
    }
    
    scores = {
        "Tích cực": 0.0,
        "Tiêu cực": 0.0,
        "Trung tính": 0.0
    }
    
    # Nếu có all_scores từ model, sử dụng scores thực
    if all_scores:
        for model_label, score in all_scores.items():
            model_label_upper = model_label.upper()
            # Tìm label tiếng Việt tương ứng
            for key, vietnamese_label in label_mapping.items():
                if key in model_label_upper:
                    scores[vietnamese_label] = score
                    break
    else:
        # Fallback: Sử dụng logic cũ nếu không có all_scores
        emotion, _ = map_sentiment_label(sentiment_label)
        scores[emotion] = confidence
        
        # Phân bổ phần còn lại cho các cảm xúc khác
        remaining = (1.0 - confidence) / 2
        for key in scores:
            if key != emotion:
                scores[key] = remaining
    
    return scores

def save_result_to_database(text, sentiment, confidence, timestamp):
    """
    Lưu kết quả vào database
    """
    insert_sentiment_analysis(text, sentiment, confidence, timestamp)

def get_result_from_database():
    """
    Lấy kết quả từ database
    """
    return get_sentiment_analysis()

with tab1:
    st.header("📝 Nhập văn bản")
    text_input = st.text_area(
        "Nhập văn bản cần phân loại cảm xúc:",
        height=200,
        placeholder="Ví dụ: Hôm nay là một ngày tuyệt vời! Tôi cảm thấy rất hạnh phúc và vui mừng.",
        key="text_input"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        classify_button = st.button("🔍 Phân loại cảm xúc", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("🗑️ Xóa", use_container_width=True)
    
    if clear_button:
        st.rerun()

    # Kết quả phân loại
    if classify_button and text_input:
        # Validate văn bản đầu vào
        is_valid, error_message = validate_text(text_input)
        
        if not is_valid:
            st.error(error_message)
        else:
            # Load analyzer nếu chưa có
            if st.session_state.analyzer is None:
                st.session_state.analyzer = load_sentiment_analyzer()
            
            if st.session_state.analyzer is None:
                st.error("❌ Không thể load sentiment analyzer. Vui lòng kiểm tra lại cài đặt!")
            else:
                with st.spinner("Đang phân loại cảm xúc..."):
                    result = classify_emotion(text_input)
                
                if result:
                    # Lấy thông tin từ kết quả (đầy đủ các trường)
                    sentiment_label = result['sentiment']
                    confidence = result.get('confidence', 1.0)
                    cleaned_text = result.get('text', text_input)
                    original_text = result.get('original_text', text_input)
                    all_scores = result.get('all_scores', None)
                    
                    # Map sentiment label sang tiếng Việt
                    emotion, emoji = map_sentiment_label(sentiment_label)
                    
                    # Tạo scores dictionary (sử dụng scores thực nếu có)
                    scores = create_scores_dict(sentiment_label, confidence, all_scores)

                    st.markdown("---")
                    st.header("🎯 Kết quả phân loại")
                    
                    # Hiển thị cảm xúc chính
                    emotion_class = "positive" if emotion == "Tích cực" else "negative" if emotion == "Tiêu cực" else "neutral"
                    st.markdown(
                        f'<div class="emotion-box {emotion_class}">{emoji} {emotion}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Hiển thị confidence
                    st.metric("Độ tin cậy", f"{confidence:.1%}")
                    
                    # Hiển thị điểm số chi tiết
                    st.subheader("📈 Điểm số chi tiết")
                    cols = st.columns(3)
                    
                    emotions_list = ["Tích cực", "Tiêu cực", "Trung tính"]
                    emojis_list = ["😊", "😢", "😐"]
                    
                    for idx, (emotion_name, emoji_icon) in enumerate(zip(emotions_list, emojis_list)):
                        with cols[idx]:
                            score = scores.get(emotion_name, 0)
                            st.metric(
                                f"{emoji_icon} {emotion_name}",
                                f"{score:.1%}",
                                delta=f"{score*100:.1f}%" if emotion_name == emotion else None
                            )
                            st.progress(score)
                    
                    # Hiển thị văn bản đã phân loại
                    st.subheader("📄 Văn bản đã xử lý")
                    with st.expander("Xem chi tiết"):
                        st.text("Văn bản gốc:")
                        st.info(original_text)
                        st.text("Văn bản đã chuẩn hóa:")
                        st.success(cleaned_text)
                    
                    # Lưu kết quả vào database sqlite
                    save_result_to_database(original_text, sentiment_label, confidence, get_timestamp())

                    st.success("✅ Kết quả đã được lưu vào lịch sử!")
                else:
                    st.error("❌ Không thể phân loại văn bản. Vui lòng thử lại!")
            
    elif classify_button and not text_input:
        st.warning("⚠️ Vui lòng nhập văn bản trước khi phân loại!")

with tab2:
    st.header("📜 Lịch sử phân loại")
    
    # Lấy dữ liệu từ database
    history_data = get_result_from_database()
    
    if len(history_data) == 0:
        st.info("📭 Chưa có lịch sử phân loại nào. Hãy phân loại một văn bản để bắt đầu!")
    else:
        # Thống kê tổng quan
        st.subheader("📊 Thống kê tổng quan")
        total_count = len(history_data)
        
        # Đếm theo sentiment label (row format: id, text, sentiment, confidence, timestamp)
        positive_count = sum(1 for row in history_data if "POS" in str(row[2]).upper())
        negative_count = sum(1 for row in history_data if "NEG" in str(row[2]).upper())
        neutral_count = total_count - positive_count - negative_count
        
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Tổng số", total_count)
        with stat_cols[1]:
            st.metric("😊 Tích cực", positive_count)
        with stat_cols[2]:
            st.metric("😢 Tiêu cực", negative_count)
        with stat_cols[3]:
            st.metric("😐 Trung tính", neutral_count)
         
        # Hiển thị lịch sử
        st.subheader(f"📋 Danh sách ({total_count} mục)")
        
        for idx, row in enumerate(history_data):
            # Format: (id, text, sentiment, confidence, timestamp)
            analysis_id, text, sentiment_label, confidence, timestamp = row
            
            # Map sentiment label sang tiếng Việt
            emotion, emoji = map_sentiment_label(sentiment_label)
            
            with st.expander(f"{emoji} {emotion} - {timestamp}", expanded=(idx == 0)):
                # Thông tin cảm xúc
                emotion_class = "positive" if emotion == "Tích cực" else "negative" if emotion == "Tiêu cực" else "neutral"
                st.markdown(
                    f'<div class="emotion-box {emotion_class}">{emoji} {emotion}</div>',
                    unsafe_allow_html=True
                )
                
                # Văn bản
                st.markdown("**📄 Văn bản:**")
                st.text_area("", text, height=100, key=f"text_{analysis_id}", disabled=True, label_visibility="collapsed")
                
                # Độ tin cậy
                st.markdown("**📈 Độ tin cậy:**")
                st.metric("Confidence", f"{confidence:.1%}")
                st.progress(confidence)
                
                # Thông tin chi tiết
                st.markdown("**ℹ️ Thông tin:**")
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.caption(f"**Sentiment Label:** {sentiment_label}")
                with col_info2:
                    st.caption(f"**⏰ Thời gian:** {timestamp}")
        

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Ứng dụng Phân Loại Cảm Xúc Văn Bản | Powered by Streamlit"
    "</div>",
    unsafe_allow_html=True
)