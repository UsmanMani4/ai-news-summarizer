import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def clean_text(text):
    """
    Basic cleaning for news articles
    """
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces
    text = re.sub(r'<.*?>', '', text)         # remove HTML tags
    text = text.strip()
    return text

def sentence_segment(text):
    return sent_tokenize(text)