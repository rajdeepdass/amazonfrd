from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import pandas as pd

def extract_features(df):
    print("Extracting text features using TF-IDF...")
    # Word-level TF-IDF
    word_vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=5,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    X_word = word_vectorizer.fit_transform(df['processed_review'])
    
    # Character n-gram features
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 5),
        max_features=2000,
        min_df=5,
        max_df=0.9
    )
    X_char = char_vectorizer.fit_transform(df['processed_review'])
    
    print("Combining with engineered features...")
    # Select the engineered features - exclude text columns, labels, and non-numeric columns
    exclude_cols = ['DOC_ID', 'LABEL', 'PRODUCT_ID', 'PRODUCT_TITLE', 'REVIEW_TITLE', 
                    'REVIEW_TEXT', 'processed_review', 'PRODUCT_CATEGORY', 'VERIFIED_PURCHASE']
    
    # Only select numeric columns
    feature_columns = []
    for col in df.columns:
        if col not in exclude_cols:
            # Try to convert to numeric, skip if not possible
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Only include if column has valid numeric values
                if not df[col].isna().all():
                    feature_columns.append(col)
            except:
                print(f"Skipping non-numeric column: {col}")
    
    print(f"Using {len(feature_columns)} numeric features: {feature_columns}")
    
    # Convert engineered features to a normalized numpy array
    X_features = df[feature_columns].fillna(0).values
    
    # Normalize engineered features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    
    # Convert to sparse matrix for efficient concatenation
    from scipy.sparse import csr_matrix
    X_features_sparse = csr_matrix(X_features)
    
    # Combine all features
    X = hstack([X_word, X_char, X_features_sparse])
    
    print("Processing labels...")
    # Check which column contains the labels
    if 'LABEL' in df.columns:
        # Map the labels if they are in the format '__label1__' and '__label2__'
        if df['LABEL'].iloc[0] is not None and '__label' in str(df['LABEL'].iloc[0]):
            y = df['LABEL'].map({'__label1__': 0, '__label2__': 1})
        else:
            # If labels are already numeric or in another format
            y = df['LABEL']
    else:
        raise KeyError("Could not find LABEL column. Available columns: " + str(df.columns.tolist()))
    
    print(f"Feature extraction complete. X shape: {X.shape}")
    return X, y, (word_vectorizer, char_vectorizer)
