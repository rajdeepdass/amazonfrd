from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import SGDClassifier

def split_data(X, y):
    # Using 90% of data for training, 10% for testing
    return train_test_split(X, y, test_size=0.1, random_state=32, shuffle=True)

def train_models(X_train, y_train):
    models = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'SVC': SVC(probability=True),
        'NuSVC': NuSVC(probability=True),
        'SGD': SGDClassifier(max_iter=1000, tol=1e-3)
    }
    
    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
    
    return models
