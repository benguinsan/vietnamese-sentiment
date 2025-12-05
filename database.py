import sqlite3
import os
from datetime import datetime

# Đường dẫn database
DB_PATH = 'sentiment_analysis.db'

def get_connection():
    """
    Tạo và trả về connection đến database
    
    Returns:
        sqlite3.Connection: Connection object
    """
    conn = sqlite3.connect(DB_PATH)
    return conn

def get_cursor(conn=None):
    """
    Lấy cursor từ connection
    
    Args:
        conn: Connection object (nếu None sẽ tạo connection mới)
    
    Returns:
        sqlite3.Cursor: Cursor object
    """
    if conn is None:
        conn = get_connection()
    return conn.cursor()

def init_database():
    """
    Khởi tạo database và tạo bảng nếu chưa tồn tại
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Tạo bảng sentiment_analysis
    cur.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def get_timestamp():
    """
    Lấy timestamp hiện tại dưới dạng YYYY-MM-DD HH:MM:SS
    
    Returns:
        str: Timestamp string theo định dạng YYYY-MM-DD HH:MM:SS
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def insert_sentiment_analysis(text, sentiment, confidence, timestamp=None):
    """
    Lưu văn bản và kết quả phân tích vào database
    
    Args:
        text: Văn bản gốc
        sentiment: Label sentiment (NEG/POS/NEU)
        confidence: Độ tin cậy (0-1)
        timestamp: Timestamp (optional, tự động tạo nếu None)
    """
    if timestamp is None:
        timestamp = get_timestamp()
    
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''INSERT INTO sentiment_analysis (text, sentiment, confidence, timestamp) 
                   VALUES (?, ?, ?, ?)''', 
                (text, sentiment, confidence, timestamp))
    conn.commit()
    conn.close()

def get_sentiment_analysis():
    """
    Lấy tất cả kết quả phân tích từ database (sắp xếp theo thời gian mới nhất)

    Returns:
        list: Danh sách các kết quả phân tích dạng tuple (id, text, sentiment, confidence, timestamp)
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''SELECT * FROM sentiment_analysis ORDER BY timestamp DESC''')
    results = cur.fetchall()
    conn.close()
    return results


# def delete_sentiment_analysis(analysis_id):
#     """
#     Xóa một bản ghi phân tích theo ID
    
#     Args:
#         analysis_id: ID của bản ghi cần xóa
#     """
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute('''DELETE FROM sentiment_analysis WHERE id = ?''', (analysis_id,))
#     conn.commit()
#     conn.close()


# def delete_all_sentiment_analysis():
#     """
#     Xóa tất cả bản ghi phân tích
#     """
#     conn = get_connection()
#     cur = conn.cursor()
#     cur.execute('''DELETE FROM sentiment_analysis''')
#     conn.commit()
#     conn.close()

