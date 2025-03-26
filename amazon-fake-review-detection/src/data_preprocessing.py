import pandas as pd
import nltk
import os
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources with offline fallback
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("NLTK resources downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("Using offline mode if data is already available.")

download_nltk_data()

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text):
    # Handle NaN values
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and stem
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback if stopwords aren't available
        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                         "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                         'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                         'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                         'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                         'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 
                         'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
                         'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                         'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                         'about', 'against', 'between', 'into', 'through', 'during', 'before', 
                         'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
                         'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                         'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 
                         'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
                         'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
                         'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
                         'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
                         "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 
                         'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
                         "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
                         'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
                         "won't", 'wouldn', "wouldn't"])
    
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def get_sentiment_scores(text):
    """Get sentiment scores with error handling"""
    try:
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(str(text))
        return scores['compound'], scores['pos'], scores['neg']
    except Exception as e:
        # Fallback if VADER isn't available
        print(f"Error in sentiment analysis: {e}")
        # Simple rule-based fallback
        text = str(text).lower()
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'poor', 'horrible']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        total = pos_count + neg_count
        
        if total == 0:
            return 0, 0, 0
        
        compound = (pos_count - neg_count) / total
        pos = pos_count / total if total > 0 else 0
        neg = neg_count / total if total > 0 else 0
        
        return compound, pos, neg

def add_engineered_features(df):
    print("  - Adding text length features...")
    # Basic features
    df['review_length'] = df['REVIEW_TEXT'].fillna('').astype(str).apply(len)
    df['capital_letters'] = df['REVIEW_TEXT'].fillna('').astype(str).apply(lambda x: sum(1 for c in x if c.isupper()))
    df['punctuation_count'] = df['REVIEW_TEXT'].fillna('').astype(str).apply(lambda x: sum(1 for c in x if c in string.punctuation))
    df['exclamation_count'] = df['REVIEW_TEXT'].fillna('').astype(str).apply(lambda x: x.count('!'))
    df['question_count'] = df['REVIEW_TEXT'].fillna('').astype(str).apply(lambda x: x.count('?'))
    
    print("  - Adding word-based features...")
    # Word-based features
    df['word_count'] = df['REVIEW_TEXT'].fillna('').astype(str).apply(lambda x: len(x.split()))
    df['avg_word_length'] = df['REVIEW_TEXT'].fillna('').astype(str).apply(
        lambda x: sum(len(word) for word in x.split()) / max(len(x.split()), 1)
    )
    
    # Ratio features
    df['capital_ratio'] = df['capital_letters'] / df['review_length'].apply(lambda x: max(x, 1))
    df['punctuation_ratio'] = df['punctuation_count'] / df['review_length'].apply(lambda x: max(x, 1))
    
    print("  - Adding sentiment features...")
    # Apply sentiment analysis with vectorized operation
    sentiment_results = df['REVIEW_TEXT'].fillna('').astype(str).apply(get_sentiment_scores)
    df['vader_compound'] = sentiment_results.apply(lambda x: x[0])
    df['vader_pos'] = sentiment_results.apply(lambda x: x[1])
    df['vader_neg'] = sentiment_results.apply(lambda x: x[2])
    
    # Title features if available
    if 'REVIEW_TITLE' in df.columns:
        print("  - Adding title features...")
        df['title_length'] = df['REVIEW_TITLE'].fillna('').astype(str).apply(len)
        df['title_word_count'] = df['REVIEW_TITLE'].fillna('').astype(str).apply(lambda x: len(x.split()))
        
        title_sentiment = df['REVIEW_TITLE'].fillna('').astype(str).apply(get_sentiment_scores)
        df['title_sentiment'] = title_sentiment.apply(lambda x: x[0])
    
    # Rating-sentiment mismatch (if rating is available)
    if 'RATING' in df.columns:
        print("  - Adding rating-sentiment features...")
        # Convert rating to numeric if it's not already
        df['RATING'] = pd.to_numeric(df['RATING'], errors='coerce').fillna(3)
        # Normalize rating to [-1, 1] scale for comparison with sentiment
        df['normalized_rating'] = (df['RATING'] - 3) / 2
        df['rating_sentiment_mismatch'] = abs(df['normalized_rating'] - df['vader_compound'])
    
    # Verified purchase feature (if available)
    if 'VERIFIED_PURCHASE' in df.columns:
        print("  - Adding verified purchase features...")
        # Convert to numeric
        df['verified_purchase_num'] = df['VERIFIED_PURCHASE'].astype(str).map({'Y': 1, 'N': 0, 'y': 1, 'n': 0}).fillna(0)
    
    return df

def prepare_data(df):
    print("Preprocessing text...")
    df['processed_review'] = df['REVIEW_TEXT'].apply(preprocess_text)
    
    print("Adding engineered features...")
    df = add_engineered_features(df)
    
    print("Data preparation complete.")
    return df
