from flask import Flask, request, render_template, jsonify
from predict import predict_review
import numpy as np
import json

app = Flask(__name__)

# Create a custom JSON encoder that handles NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Configure Flask to use the custom encoder
app.json.encoder = NumpyEncoder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        review_text = request.form.get('review', '')
        review_title = request.form.get('title', '')
        rating = request.form.get('rating')
        verified = request.form.get('verified')
        
        # Convert rating to float if provided
        if rating:
            try:
                rating = float(rating)
            except:
                rating = None
        
        # Convert verified to proper format
        if verified == 'Yes':
            verified = 'Y'
        elif verified == 'No':
            verified = 'N'
        
        results = predict_review(
            review_text=review_text,
            review_title=review_title,
            rating=rating,
            verified_purchase=verified
        )
        
        # Ensure all numpy values are converted to Python native types
        for model_name in results:
            if 'confidence' in results[model_name] and results[model_name]['confidence'] is not None:
                results[model_name]['confidence'] = float(results[model_name]['confidence'])
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
