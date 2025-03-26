from sklearn.metrics import classification_report, accuracy_score

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name} model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        results[name] = {'accuracy': accuracy, 'report': report}
    return results
