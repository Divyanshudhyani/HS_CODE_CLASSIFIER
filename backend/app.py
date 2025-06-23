from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# --- Improved Path Handling and Model/Data Loading ---
try:
    # Get the directory where this script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Construct relative paths
    model_path = os.path.join(BASE_DIR, 'model', 'model.pkl')
    data_path = os.path.join(BASE_DIR, '..', 'data', 'harmonized-stem.csv')

    # Load the model and data
    model = joblib.load(model_path)
    hs_df = pd.read_csv(data_path)
    
    # Create a lookup dictionary from HS code to product name
    # Ensures keys are strings, matching the model's output format
    hs_code_to_name = hs_df.set_index(hs_df['hs_product_code'].astype(str))['hs_product_name_short_en'].to_dict()
    
    print("✅ Model and data loaded successfully.")
    
except Exception as e:
    print(f"❌ Error loading model or data: {e}")
    model = None
    hs_code_to_name = {}

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        desc = data.get('description', '')
        if not desc.strip():
            return jsonify({"error": "Empty description"}), 400

        # --- Optimized Prediction and Response ---
        # Predict probabilities to get both prediction and confidence
        probabilities = model.predict_proba([desc])[0]
        max_prob_index = np.argmax(probabilities)
        
        prediction = model.classes_[max_prob_index]
        confidence = probabilities[max_prob_index]
        category = hs_code_to_name.get(str(prediction), "Unknown Category")

        return jsonify({
            'predicted_hs_code': str(prediction),
            'predicted_hs_category': category,
            'confidence': round(float(confidence), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
            
        df = pd.read_csv(file)
        if 'description' not in df.columns:
            return jsonify({"error": "Missing 'description' column"}), 400

        # --- Enhanced Bulk Prediction Logic ---
        descriptions = df['description'].fillna('')
        probabilities = model.predict_proba(descriptions)
        
        df['predicted_hs_code'] = model.predict(descriptions)
        df['confidence'] = np.max(probabilities, axis=1).round(2)
        df['predicted_hs_category'] = df['predicted_hs_code'].astype(str).map(hs_code_to_name).fillna("Unknown Category")
        
        return jsonify(df.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
