from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import os
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Add custom headers with DYMRA TECH branding
@app.after_request
def add_header(response):
    response.headers['X-Powered-By'] = 'DYMRA TECH'
    response.headers['X-Application'] = 'HS Code Classifier'
    return response

# --- Improved Path Handling and Model/Data Loading ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'model', 'model_bert.pkl')
    data_path = os.path.join(BASE_DIR, '..', 'data', 'harmonized-stem.csv')

    # Load the BERT-based model and classifier
    model_data = joblib.load(model_path)
    clf = model_data['clf']
    bert_model = SentenceTransformer(model_data['bert_model'])
    hs_df = pd.read_csv(data_path)
    hs_code_to_name = hs_df.set_index(hs_df['hs_product_code'].astype(str))['hs_product_name_short_en'].to_dict()
    print("✅ BERT model and data loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or data: {e}")
    clf = None
    bert_model = None
    hs_code_to_name = {}

@app.route('/')
def home():
    return jsonify({
        "message": "HS Code Classifier API",
        "company": "DYMRA TECH",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "bulk_predict": "/bulk_predict"
        }
    })

@app.route('/logo')
def logo():
    """Return logo information"""
    return jsonify({
        "company": "DYMRA TECH",
        "logo": {
            "type": "circuit_board_network",
            "colors": {
                "background": "blue_gradient",
                "elements": "white"
            },
            "description": "Stylized circuit board with connected nodes representing technology and connectivity"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if clf is None or bert_model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        desc = data.get('description', '')
        if not desc.strip():
            return jsonify({"error": "Empty description"}), 400

        # Encode input with BERT
        desc_embedding = bert_model.encode([desc])
        probabilities = clf.predict_proba(desc_embedding)[0]
        max_prob_index = np.argmax(probabilities)
        prediction = clf.classes_[max_prob_index]
        confidence = probabilities[max_prob_index]
        category = hs_code_to_name.get(str(prediction), "Unknown Category")

        # --- Rule-based fallback for low confidence ---
        if confidence < 0.3:
            desc_lower = desc.lower().strip()
            if desc_lower == "plastic":
                prediction = "39"
                category = hs_code_to_name.get("39", "Plastics")
                confidence = 0.8
            # (other rules can go here)

        return jsonify({
            'predicted_hs_code': str(prediction),
            'predicted_hs_category': category,
            'confidence': round(float(confidence), 2),
            'powered_by': 'DYMRA TECH'
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    if clf is None or bert_model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        df = pd.read_csv(file)
        if 'description' not in df.columns:
            return jsonify({"error": "Missing 'description' column"}), 400
        descriptions = df['description'].fillna('')
        desc_embeddings = bert_model.encode(descriptions.tolist(), show_progress_bar=False)
        probabilities = clf.predict_proba(desc_embeddings)
        df['predicted_hs_code'] = clf.classes_[np.argmax(probabilities, axis=1)]
        df['confidence'] = np.max(probabilities, axis=1).round(2)
        df['predicted_hs_category'] = df['predicted_hs_code'].astype(str).map(hs_code_to_name).fillna("Unknown Category")
        
        # Add metadata to response
        results = df.to_dict(orient='records')
        return jsonify({
            'results': results,
            'total_predictions': len(results),
            'powered_by': 'DYMRA TECH'
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
