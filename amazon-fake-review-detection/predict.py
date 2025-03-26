import pandas as pd
import joblib
from src.data_preprocessing import preprocess_text, add_engineered_features
from scipy.sparse import hstack, csr_matrix
import numpy as np

def load_models(model_path='models/'):
    """Load trained models and vectorizers"""
    models = {}
    for model_name in ['XGBoost', 'SVC', 'NuSVC', 'SGD']:
        try:
            models[model_name] = joblib.load(f'{model_path}{model_name}_model.pkl')
        except Exception as e:
            print(f"Could not load {model_name} model: {e}")
    
    # Load vectorizers
    word_vectorizer = joblib.load(f'{model_path}word_vectorizer.pkl')
    char_vectorizer = joblib.load(f'{model_path}char_vectorizer.pkl')
    scaler = joblib.load(f'{model_path}feature_scaler.pkl')
    
    return models, word_vectorizer, char_vectorizer, scaler

def predict_review(review_text, review_title="", rating=None, verified_purchase=None):
    """Predict if a review is fake or genuine"""
    # Load models and vectorizers
    models, word_vectorizer, char_vectorizer, scaler = load_models()
    
    # Create a dataframe with the review
    review_data = {
        'REVIEW_TEXT': review_text,
        'REVIEW_TITLE': review_title,
        'RATING': rating if rating is not None else 3,
        'VERIFIED_PURCHASE': verified_purchase if verified_purchase is not None else 'N'
    }
    df = pd.DataFrame([review_data])
    
    # Preprocess the review
    df['processed_review'] = df['REVIEW_TEXT'].apply(preprocess_text)
    
    # Add engineered features
    df = add_engineered_features(df)
    
    # Extract features
    X_word = word_vectorizer.transform(df['processed_review'])
    X_char = char_vectorizer.transform(df['processed_review'])
    
    # Convert VERIFIED_PURCHASE to numeric
    if 'VERIFIED_PURCHASE' in df.columns:
        df['verified_purchase_num'] = df['VERIFIED_PURCHASE'].map({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0}).fillna(0)
        # Remove the original column to avoid conversion errors
        df = df.drop('VERIFIED_PURCHASE', axis=1)
    
    # Select only numeric columns for feature extraction
    exclude_cols = ['REVIEW_TEXT', 'REVIEW_TITLE', 'processed_review']
    numeric_features = []
    
    for col in df.columns:
        if col not in exclude_cols:
            # Try to convert to numeric, include if successful
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if not df[col].isna().all():  # Only include if column has valid values
                    numeric_features.append(col)
            except:
                pass  # Skip non-numeric columns
    
    # Convert engineered features to a normalized numpy array
    X_features = df[numeric_features].fillna(0).values
    
    # Normalize engineered features
    X_features = scaler.transform(X_features)
    
    # Convert to sparse matrix for efficient concatenation
    X_features_sparse = csr_matrix(X_features)
    
    # Combine all features
    X = hstack([X_word, X_char, X_features_sparse])
    
    # Make predictions with each model
    results = {}
    for name, model in models.items():
        try:
            # Get prediction (0 = genuine, 1 = fake)
            prediction = model.predict(X)[0]
            
            # Get probability scores if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                confidence = proba[1] if prediction == 1 else proba[0]
            else:
                confidence = None
                
            results[name] = {
                'prediction': 'FAKE' if prediction == 0 else 'GENUINE',
                'confidence': confidence
            }
        except Exception as e:
            results[name] = {'prediction': 'ERROR', 'confidence': None, 'error': str(e)}
    
    return results
