from src.data_preprocessing import load_data, prepare_data
from src.feature_extraction import extract_features
from src.model_training import split_data, train_models
from src.evaluation import evaluate_models
import os
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

def main():
    # Load and preprocess data
    print("Loading data...")
    df = load_data('data/amazon_reviews.csv')
    df = prepare_data(df)
    print(df.columns)

    # Extract features
    X, y, vectorizers = extract_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models
    models = train_models(X_train, y_train)

    # Evaluate models
    results = evaluate_models(models, X_test, y_test)

    # Print results
    for name, result in results.items():
        print(f"\nResults for {name}:")
        print(f"Accuracy: {result['accuracy']}")
        print("Classification Report:")
        print(result['report'])

    # Save models and vectorizers for later use
    print("\nSaving models and vectorizers...")
    os.makedirs('models', exist_ok=True)
    
    # Save models
    for name, model in models.items():
        joblib.dump(model, f'models/{name}_model.pkl')
    
    # Save vectorizers
    word_vectorizer, char_vectorizer = vectorizers
    joblib.dump(word_vectorizer, 'models/word_vectorizer.pkl')
    joblib.dump(char_vectorizer, 'models/char_vectorizer.pkl')
    
    # Save scaler - only use numeric columns
    exclude_cols = ['DOC_ID', 'LABEL', 'PRODUCT_ID', 'PRODUCT_TITLE', 'REVIEW_TITLE', 
                    'REVIEW_TEXT', 'processed_review', 'PRODUCT_CATEGORY', 'VERIFIED_PURCHASE']
    
    # Only select numeric columns
    feature_columns = [col for col in df.columns if col not in exclude_cols and 
                      pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"Using {len(feature_columns)} numeric features for scaler")
    
    scaler = StandardScaler()
    scaler.fit(df[feature_columns].fillna(0).values)
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    print("Training and evaluation complete. Models saved to 'models/' directory.")

if __name__ == "__main__":
    main()
