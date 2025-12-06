# Ứng dụng Phân Loại Cảm Xúc Văn Bản Tiếng Việt

Ứng dụng web Streamlit để phân tích và phân loại cảm xúc trong văn bản tiếng Việt sử dụng mô hình PhoBERT đã được fine-tune.

## ✨ Tính năng

- 🔍 **Phân tích cảm xúc tự động**: Phân loại văn bản thành Tích cực, Tiêu cực, hoặc Trung tính
- 📊 **Hiển thị điểm số chi tiết**: Xem confidence score và phân bổ điểm số cho từng loại cảm xúc
- 📜 **Lịch sử phân tích**: Lưu trữ và xem lại tất cả các kết quả phân tích trong database SQLite
- 🔤 **Xử lý văn bản tiếng Việt**: 
  - Tự động restore dấu cho văn bản không dấu sử dụng model `peterhung/vietnamese-accent-marker-xlm-roberta`
  - Chuẩn hóa văn bản với thư viện `underthesea` (xử lý từ viết tắt, emoticon, từ viết liền, word tokenization)

## 🚀 Hướng dẫn cài đặt và sử dụng

### Yêu cầu hệ thống

- **Python**: 3.10 trở lên
- **pip**: Phiên bản mới nhất
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+ để load models)
- **Dung lượng ổ cứng**: ~2GB để lưu models từ HuggingFace
- **Kết nối Internet**: Cần thiết cho lần chạy đầu tiên để tải models

### Các bước cài đặt

#### Bước 1: Clone hoặc tải project về máy

```bash
# Nếu có Git repository
git clone <https://github.com/benguinsan/vietnamese-sentiment>

# Hoặc giải nén file ZIP vào thư mục Seminar
```

#### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
# Tạo virtual environment
python3.11 -m venv venv

# Kích hoạt virtual environment
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```

#### Bước 3: Cài đặt các thư viện cần thiết

```bash
# Cài đặt tất cả dependencies từ requirements.txt
pip3.11 install -r requirements.txt
```

**Các thư viện chính được sử dụng**:
- `streamlit==1.51.0`: Framework web application
- `torch==2.9.1`: Framework deep learning (PyTorch)
- `transformers==4.57.1`: Thư viện HuggingFace để sử dụng pre-trained models
- `underthesea==8.3.0`: Thư viện NLP tiếng Việt (text normalization, word tokenization)
- `numpy==2.3.5`: Thư viện tính toán số học
- `pandas==2.3.3`: Xử lý dữ liệu (nếu cần)

#### Bước 4: Kiểm tra file cần thiết

Đảm bảo các file sau có trong thư mục project:
- ✅ `app.py` - File chính của ứng dụng
- ✅ `vietnamese_sentiment.py` - Module xử lý sentiment analysis
- ✅ `database.py` - Module quản lý database
- ✅ `selected_tags_names.txt` - File tags cho diacritic restoration (quan trọng!)

### Chạy ứng dụng

#### Lần đầu chạy

```bash
streamlit run app.py
```

**Lưu ý quan trọng**:
- Lần đầu chạy, ứng dụng sẽ tự động tải 2 models từ HuggingFace:
  - `wonrax/phobert-base-vietnamese-sentiment` (~500MB)
  - `peterhung/vietnamese-accent-marker-xlm-roberta` (~1GB)
- Quá trình tải có thể mất **5-15 phút** tùy tốc độ internet
- Models sẽ được cache trong thư mục `~/.cache/huggingface/` để sử dụng cho các lần sau

#### Sau khi models đã được tải

Ứng dụng sẽ tự động mở trong trình duyệt tại địa chỉ:
```
http://localhost:8501
```

Nếu không tự động mở, bạn có thể truy cập thủ công bằng cách copy địa chỉ này vào trình duyệt.

#### Dừng ứng dụng

Nhấn `Ctrl + C` trong terminal để dừng server Streamlit.

### Xử lý lỗi thường gặp

#### Lỗi: "Các thư viện ML chưa được cài đặt"

**Nguyên nhân**: Thiếu các thư viện `torch`, `transformers`, hoặc `underthesea`

**Giải pháp**:
```bash
pip install torch transformers underthesea numpy
```

#### Lỗi: "FileNotFoundError: selected_tags_names.txt"

**Nguyên nhân**: File `selected_tags_names.txt` không có trong thư mục project

**Giải pháp**: Đảm bảo file `selected_tags_names.txt` nằm cùng thư mục với `vietnamese_sentiment.py`

#### Lỗi: "CUDA out of memory" hoặc chạy chậm

**Nguyên nhân**: Model quá lớn cho GPU/RAM hiện tại

**Giải pháp**: 
- Models sẽ tự động sử dụng CPU nếu không có GPU
- Nếu RAM không đủ, đóng các ứng dụng khác hoặc sử dụng máy có RAM lớn hơn

#### Lỗi: "Connection timeout" khi tải models

**Nguyên nhân**: Kết nối internet không ổn định

**Giải pháp**: 
- Kiểm tra kết nối internet
- Chạy lại ứng dụng, models sẽ tiếp tục tải từ điểm dừng
- Hoặc tải models thủ công từ HuggingFace và đặt vào cache folder

#### Lỗi: "Port 8501 is already in use"

**Nguyên nhân**: Đã có một instance Streamlit đang chạy

**Giải pháp**:
```bash
# Tìm và kill process đang dùng port 8501
# Trên macOS/Linux:
lsof -ti:8501 | xargs kill -9

# Trên Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Hoặc chạy trên port khác:
streamlit run app.py --server.port 8502
```

## 📁 Cấu trúc dự án

```
Seminar/
├── app.py                      # File chính của ứng dụng Streamlit
├── vietnamese_sentiment.py     # Module xử lý sentiment analysis
│   ├── VietnameseDiacriticRestorer    # Restore dấu tiếng Việt
│   ├── VietnameseTextStandardizer     # Chuẩn hóa văn bản
│   └── VietnameseSentimentAnalyzer    # Phân tích sentiment
├── database.py                 # Module quản lý database SQLite
│   ├── init_database()        # Khởi tạo database
│   ├── insert_sentiment_analysis()    # Lưu kết quả phân tích
│   └── get_sentiment_analysis()       # Lấy lịch sử phân tích
├── requirements.txt            # Dependencies
├── selected_tags_names.txt     # File tags cho accent restoration
├── sentiment_analysis.db      # Database SQLite (tự động tạo)
├── README.md                   # File này

```

## 🔧 Hướng dẫn sử dụng chi tiết

### Giao diện chính

Khi mở ứng dụng, bạn sẽ thấy:
- **Header**: "Trợ lý phân loại cảm xúc văn bản tiếng Việt"
- **Sidebar bên trái**: Thông tin về ứng dụng và các loại cảm xúc
- **2 Tabs chính**: "🔍 Phân tích" và "📜 Lịch sử"

### Tab "🔍 Phân tích" - Phân tích cảm xúc

#### Bước 1: Nhập văn bản

1. Click vào tab **"🔍 Phân tích"** (mặc định khi mở ứng dụng)
2. Trong ô text area có nhãn **"Nhập văn bản cần phân tích cảm xúc:"**
3. Nhập hoặc paste văn bản tiếng Việt cần phân tích

**Ví dụ văn bản hợp lệ**:
- ✅ "Hôm nay tôi rất vui và hạnh phúc!"
- ✅ "Sản phẩm này không tốt, tôi thất vọng."
- ✅ "Hôm nay trời mưa. Tôi đi làm như bình thường."

**Văn bản không hợp lệ** (sẽ bị từ chối):
- ❌ Văn bản rỗng
- ❌ Ít hơn 5 ký tự (không tính khoảng trắng)
- ❌ Chỉ chứa ký tự lặp lại (ví dụ: "aaaaaa")
- ❌ Không chứa chữ cái
- ❌ Quá nhiều ký tự đặc biệt (>50% ký tự)

#### Bước 2: Phân tích

1. Click nút **"🔍 Phân loại cảm xúc"** (nút màu đỏ/primary)
2. Đợi vài giây để hệ thống xử lý:
   - Restore dấu (nếu văn bản không dấu)
   - Chuẩn hóa văn bản
   - Phân tích sentiment

#### Bước 3: Xem kết quả

Sau khi phân loại, bạn sẽ thấy:

**a) Kết quả chính**:
- **Hộp cảm xúc lớn**: Hiển thị loại cảm xúc được dự đoán
  - 😊 **Tích cực** (màu xanh lá)
  - 😢 **Tiêu cực** (màu đỏ)
  - 😐 **Trung tính** (màu xanh dương)

**b) Độ tin cậy**:
- Metric hiển thị phần trăm confidence (ví dụ: 98.7%)
- Càng cao càng chính xác

**c) Điểm số chi tiết**:
- 3 metrics cho từng loại cảm xúc:
  - 😊 Tích cực: X%
  - 😢 Tiêu cực: Y%
  - 😐 Trung tính: Z%
- Progress bar cho mỗi loại

**d) Văn bản đã xử lý** (trong expander "Xem chi tiết"):
- **Văn bản gốc**: Văn bản bạn nhập vào
- **Văn bản đã chuẩn hóa**: Văn bản sau khi restore dấu và chuẩn hóa

**e) Thông báo lưu**:
- ✅ "Kết quả đã được lưu vào lịch sử!" - Kết quả đã được lưu vào database

#### Nút "🗑️ Xóa"

- Click để xóa nội dung trong ô text area và làm mới trang

### Tab "📜 Lịch sử" - Xem lịch sử phân tích

#### Thống kê tổng quan

Ở đầu tab, bạn sẽ thấy 4 metrics:
- **Tổng số**: Tổng số lần phân tích
- **😊 Tích cực**: Số lần phân tích cho kết quả tích cực
- **😢 Tiêu cực**: Số lần phân tích cho kết quả tiêu cực
- **😐 Trung tính**: Số lần phân tích cho kết quả trung tính

#### Danh sách lịch sử

- Mỗi bản ghi được hiển thị trong một **expander**
- Format: `{emoji} {cảm xúc} - {timestamp}`
- Bản ghi mới nhất được mở sẵn (expanded)

**Trong mỗi bản ghi, bạn có thể xem**:
- **Hộp cảm xúc**: Loại cảm xúc với màu tương ứng
- **Văn bản**: Text area hiển thị văn bản đã phân tích (read-only)
- **Độ tin cậy**: Metric và progress bar
- **Thông tin chi tiết**:
  - Sentiment Label: NEG/POS/NEU
  - ⏰ Thời gian: Timestamp đầy đủ

**Lưu ý**: 
- Lịch sử được sắp xếp theo thời gian mới nhất trước
- Tất cả kết quả được lưu vĩnh viễn trong database SQLite

### Ví dụ sử dụng

#### Ví dụ 1: Văn bản tích cực
```
Input: "hôm nay tôi rất vui"
Kết quả: 😊 Tích cực (98.7%)
```

#### Ví dụ 2: Văn bản tiêu cực
```
Input: "toi cam thay rat buon va that vong"
Kết quả: 😢 Tiêu cực (95.2%)
```

#### Ví dụ 3: Văn bản trung tính
```
Input: "Hôm nay trời mưa. Tôi đi làm như bình thường."
Kết quả: 😐 Trung tính (87.3%)
```

## 🛡️ Validation

Ứng dụng tự động kiểm tra và từ chối các văn bản:

- ❌ Văn bản rỗng
- ❌ Ít hơn 5 ký tự (không tính khoảng trắng)
- ❌ Chỉ chứa ký tự lặp lại (ví dụ: "aaaaaa")
- ❌ Không chứa chữ cái
- ❌ Quá nhiều ký tự đặc biệt (>50%)

## 🗄️ Database

Ứng dụng sử dụng SQLite để lưu trữ lịch sử phân tích. Database tự động được tạo khi chạy ứng dụng lần đầu.

**Schema**:
- `id`: INTEGER PRIMARY KEY
- `text`: TEXT - Văn bản gốc
- `sentiment`: TEXT - Label sentiment (NEG/POS/NEU)
- `confidence`: REAL - Độ tin cậy (0-1)
- `timestamp`: TEXT - Thời gian phân tích (YYYY-MM-DD HH:MM:SS)

## 🤖 Models và Thư viện NLP

### Models từ HuggingFace

1. **PhoBERT** (`wonrax/phobert-base-vietnamese-sentiment`)
   - Model chính để phân tích sentiment
   - Tự động tải xuống khi chạy ứng dụng lần đầu

2. **Vietnamese Accent Marker** (`peterhung/vietnamese-accent-marker-xlm-roberta`)
   - Model để restore dấu tiếng Việt cho văn bản không dấu
   - Sử dụng XLM-RoBERTa architecture
   - Tự động tải xuống khi chạy ứng dụng lần đầu

### Thư viện xử lý văn bản

- **underthesea**: Thư viện NLP tiếng Việt
  - `text_normalize()`: Chuẩn hóa văn bản (unicode, lowercase)
  - `word_tokenize()`: Tách từ tiếng Việt

Tất cả models và thư viện tự động được tải xuống khi chạy ứng dụng lần đầu (cần kết nối internet).

## 📝 Ghi chú

- Lần đầu chạy ứng dụng, model sẽ được tải xuống (có thể mất vài phút)
- Database được tạo tự động trong thư mục gốc của project
- Tất cả kết quả phân tích được lưu vĩnh viễn trong database

## 🔗 Tài liệu tham khảo

- [Streamlit Documentation](https://docs.streamlit.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PhoBERT Model](https://huggingface.co/wonrax/phobert-base-vietnamese-sentiment)
- [Vietnamese Accent Marker Model](https://huggingface.co/peterhung/vietnamese-accent-marker-xlm-roberta)
- [underthesea Documentation](https://github.com/undertheseanlp/underthesea)

## 📄 License

Dự án này được phát triển cho mục đích học tập và nghiên cứu.
