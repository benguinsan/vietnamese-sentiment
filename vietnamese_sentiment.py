"""
Module xử lý sentiment analysis cho tiếng Việt
Chứa các class: VietnameseDiacriticRestorer, VietnameseTextStandardizer, VietnameseSentimentAnalyzer
"""

import os
import re
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    AutoModelForTokenClassification
)
from underthesea import text_normalize, word_tokenize

# Đường dẫn file tags (tự động tìm trong cùng thư mục)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAGS_FILE = os.path.join(BASE_DIR, "selected_tags_names.txt")


class VietnameseDiacriticRestorer:
    """Class để restore dấu tiếng Việt cho text không dấu"""
    
    def __init__(self, model_path='peterhung/vietnamese-accent-marker-xlm-roberta'):
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
        self.TOKENIZER_WORD_PREFIX = "▁"

        # Device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        # Load labels list
        self.label_list = self._load_tags_set(TAGS_FILE)

    def insert_accents(self, text):
        """Insert accents vào text"""
        our_tokens = text.strip().split()

        # The tokenizer may further split our tokens
        inputs = self.tokenizer(
            our_tokens,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        input_ids = inputs['input_ids']
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = tokens[1:-1]

        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

        predictions = outputs["logits"].cpu().numpy()
        predictions = np.argmax(predictions, axis=2)

        # Exclude output at index 0 and the last index, which correspond to '<s>' and '</s>'
        predictions = predictions[0][1:-1]

        return tokens, predictions

    def _load_tags_set(self, fpath):
        """Load tags từ file"""
        labels = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
        return labels

    def merge_tokens_and_preds(self, tokens, predictions):
        """Merge tokens và predictions"""
        merged_tokens_preds = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            label_indexes = set([predictions[i]])
            if tok.startswith(self.TOKENIZER_WORD_PREFIX):  # Start a new word
                tok_no_prefix = tok[len(self.TOKENIZER_WORD_PREFIX):]
                cur_word_toks = [tok_no_prefix]
                # Check if subsequent toks are part of this word
                j = i + 1
                while j < len(tokens):
                    if not tokens[j].startswith(self.TOKENIZER_WORD_PREFIX):
                        cur_word_toks.append(tokens[j])
                        label_indexes.add(predictions[j])
                        j += 1
                    else:
                        break
                cur_word = ''.join(cur_word_toks)
                merged_tokens_preds.append((cur_word, label_indexes))
                i = j
            else:
                merged_tokens_preds.append((tok, label_indexes))
                i += 1

        return merged_tokens_preds

    def get_accented_words(self, merged_tokens_preds, label_list):
        """Get accented words từ merged tokens và predictions"""
        accented_words = []
        for word_raw, label_indexes in merged_tokens_preds:
            # Use the first label that changes word_raw
            word_accented = word_raw
            for label_index in label_indexes:
                tag_name = label_list[int(label_index)]
                raw, vowel = tag_name.split("-")
                if raw and raw in word_raw:
                    word_accented = word_raw.replace(raw, vowel)
                    break

            accented_words.append(word_accented)

        return " ".join(accented_words)

    def restore(self, text):
        """Restore dấu cho text"""
        tokens, predictions = self.insert_accents(text)
        merged_tokens_preds = self.merge_tokens_and_preds(tokens, predictions)
        accented_words = self.get_accented_words(merged_tokens_preds, self.label_list)
        return accented_words


class VietnameseTextStandardizer:
    """Class để chuẩn hóa text tiếng Việt"""
    
    def __init__(self):
        # Từ điển chuẩn hóa từ viết tắt/thông dụng
        self.normalization_dict = {
            "sp": "sản phẩm", "dk": "được", "dc": "được", "ko": "không",
            "k": "không", "bt": "bình thường", "ok": "tốt", "oke": "tốt",
            "okela": "tốt", "sg": "sài gòn", "hn": "hà nội", "tks": "cảm ơn",
            "thank": "cảm ơn", "please": "làm ơn", "thanks": "cảm ơn",
            "good": "tốt", "bad": "tệ", "very": "rất", "like": "thích",
            "hate": "ghét", "du": "đủ"
        }

        # Các từ viết liền nhau
        self.joined_words_dict = {
            "toithich": "tôi thích",
            "toimuon": "tôi muốn",
            "toicamthay": "tôi cảm thấy",
            "ratthich": "rất thích",
            "quathich": "quá thích",
            "thichqua": "thích quá",
            "banthat": "bạn thật",
            "spnay": "sản phẩm này",
            "dichvunay": "dịch vụ này",
            "toikhong": "tôi không",
            "toiko": "tôi không",
            "toicung": "tôi cũng",
            "toiratthich": "tôi rất thích",
        }

        # Các từ có kí hiệu emote
        self.emoticon_sentiment_dict = {
            # Positive emoticons
            ":)": " tích_cực ", ":-)": " tích_cực ", "=)": " tích_cực ",
            ":D": " rất_tích_cực ", ":-D": " rất_tích_cực ", "=D": " rất_tích_cực ",
            "😊": " tích_cực ", "😍": " rất_tích_cực ",
            "🤩": " rất_tích_cực ", "👍": " tốt ", "❤️": " yêu_thích ",
            "💖": " yêu_thích ", "😘": " yêu_thích ", "🥰": " yêu_thích ",
            "😁": " vui ", "😄": " vui ", "😆": " vui ", "😂": " vui ",

            # Negative emoticons
            ":(": " tiêu_cực ", ":-(": " tiêu_cực ", "=(": " tiêu_cực ",
            ":'(": " buồn ", "😞": " buồn ", "😔": " buồn ", "😟": " lo_lắng ",
            "😠": " tức_giận ", "😡": " rất_tức_giận ", "🤬": " rất_tức_giận ",
            "👎": " tệ ", "💔": " thất_vọng ", "😢": " khóc ", "😭": " khóc_nhiều ",

            # Neutral/Sarcastic
            ":|": " bình_thường ", ":-|": " bình_thường ", "😐": " bình_thường ",
            "😑": " không_hài_lòng ", "🤨": " nghi_ngờ ", "😒": " chán ",
            "🙄": " mắt_đảo ", "😏": " mỉa_mai "
        }

        # EMOJI PATTERNS (để không remove hoàn toàn)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE
        )

    def split_joined_words(self, text):
        """Tách từ viết liền bằng từ điển"""
        for joined_word, separated in self.joined_words_dict.items():
            text = text.replace(joined_word, separated)
        return text

    def handle_emoticons(self, text):
        """Chuyển emoticons thành sentiment words"""
        for emoticon, sentiment_word in self.emoticon_sentiment_dict.items():
            text = text.replace(emoticon, sentiment_word)
        return text

    def standardize(self, text):
        """
        Chuẩn hóa tiếng Việt toàn diện
        """
        if not text or not isinstance(text, str):
            return ""

        # Bước 1: Chuẩn hóa unicode & lowercase
        normalized = text_normalize(text)

        # Bước 2: Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', normalized).strip()

        # Bước 3: Chuẩn hóa emote
        text = self.handle_emoticons(text)

        # Bước 4: Tách từ viết liền
        text = self.split_joined_words(text)

        # Bước 5: Tách từ (QUAN TRỌNG)
        tokens = word_tokenize(text)

        # Bước 6: Chuẩn hóa từ vựng
        standardized_tokens = []
        for token in tokens:
            # Chuẩn hóa từ viết tắt/thông dụng
            standardized_token = self.normalization_dict.get(token.lower(), token.lower())
            standardized_tokens.append(standardized_token)

        # Bước 7: Ghép lại thành câu chuẩn
        clean_text = " ".join(standardized_tokens)

        return clean_text


class VietnameseSentimentAnalyzer:
    """Class chính để phân tích sentiment tiếng Việt"""
    
    def __init__(self, model_name="wonrax/phobert-base-vietnamese-sentiment"):
        """
        Khởi tạo Vietnamese Sentiment Analyzer
        
        Args:
            model_name: Tên model từ HuggingFace
                - "wonrax/phobert-base-vietnamese-sentiment" (PhoBERT sentiment) - Default
                - "vinai/phobert-base" (PhoBERT base)
                - "FPTAI/vibert-base-cased" (ViBERT)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer
        )

        # Chuẩn hóa text
        self.standardizer = VietnameseTextStandardizer()
        self.restored = VietnameseDiacriticRestorer()

    def analyze_sentiment(self, text):
        """
        Phân tích sentiment cho text tiếng Việt
        
        Args:
            text: Text cần phân tích
            
        Returns:
            dict: {
                'original_text': text gốc,
                'text': text đã được xử lý,
                'sentiment': label (NEG/POS/NEU),
                'confidence': confidence score (0-1)
                'all_scores': scores cho tất cả labels
            }
        """
        # 1. Restore dấu
        restored_text = self.restored.restore(text)

        # 2. Chuẩn hóa text
        cleaned_text = self.standardizer.standardize(restored_text)

        # 3. Phân tích sentiment bằng model đã fine-tuned
        # Lấy tất cả scores để hiển thị đầy đủ
        result = self.sentiment_pipeline(cleaned_text, return_all_scores=True)
        
        # Tạo dictionary scores cho tất cả labels
        all_scores = {}
        for item in result[0]:
            all_scores[item['label']] = item['score']
        
        # Lấy label có score cao nhất
        top_result = max(result[0], key=lambda x: x['score'])

        return {
            'original_text': text,
            'text': cleaned_text,
            'sentiment': top_result['label'],
            'confidence': top_result['score'],
            'all_scores': all_scores 
        }