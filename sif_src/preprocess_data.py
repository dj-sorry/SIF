import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text_series):

    text_series = text_series.str.lower()
    text_series = text_series.apply(word_tokenize)

    stop_words = set(stopwords.words('english'))

    text_series = text_series.apply(
        lambda tokens: [word for word in tokens if word.isalnum() and word not in stop_words]
    )
    
    return text_series
