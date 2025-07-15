from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure app
app.config['DEBUG'] = Config.DEBUG

# Add custom headers with DYMRA TECH branding
@app.after_request
def add_header(response):
    response.headers['X-Powered-By'] = 'DYMRA TECH'
    response.headers['X-Application'] = 'HS Code Classifier'
    return response

# Input validation function
def validate_description(desc):
    """Validate product description input"""
    if not desc or not desc.strip():
        raise ValueError("Description cannot be empty")
    
    desc = desc.strip()
    if len(desc) < Config.MIN_DESCRIPTION_LENGTH:
        raise ValueError(f"Description too short. Minimum {Config.MIN_DESCRIPTION_LENGTH} characters required.")
    
    if len(desc) > Config.MAX_DESCRIPTION_LENGTH:
        raise ValueError(f"Description too long. Maximum {Config.MAX_DESCRIPTION_LENGTH} characters allowed.")
    
    return desc

# --- Improved Path Handling and Model/Data Loading ---
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, '..', Config.MODEL_PATH)
    data_path = os.path.join(BASE_DIR, '..', Config.DATA_PATH)

    # Load the BERT-based model and classifier
    model_data = joblib.load(model_path)
    clf = model_data['clf']
    bert_model = SentenceTransformer(model_data['bert_model'])
    hs_df = pd.read_csv(data_path)
    hs_code_to_name = hs_df.set_index(hs_df['hs_product_code'].astype(str))['hs_product_name_short_en'].to_dict()
    logger.info("✅ BERT model and data loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model or data: {e}")
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
        },
        "config": {
            "max_description_length": Config.MAX_DESCRIPTION_LENGTH,
            "min_description_length": Config.MIN_DESCRIPTION_LENGTH,
            "low_confidence_threshold": Config.LOW_CONFIDENCE_THRESHOLD
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
        logger.error("Model not loaded")
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        desc = data.get('description', '')
        
        # Validate input
        try:
            validated_desc = validate_description(desc)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Encode input with BERT
        desc_embedding = bert_model.encode([validated_desc])
        probabilities = clf.predict_proba(desc_embedding)[0]
        max_prob_index = np.argmax(probabilities)
        prediction = clf.classes_[max_prob_index]
        confidence = probabilities[max_prob_index]
        category = hs_code_to_name.get(str(prediction), "Unknown Category")

        # --- Rule-based fallback for low confidence ---
        if confidence < Config.LOW_CONFIDENCE_THRESHOLD:
            desc_lower = validated_desc.lower().strip()
            if desc_lower == "plastic":
                prediction = "39"
                category = hs_code_to_name.get("39", "Plastics")
                confidence = 0.8
            # (other rules can go here)

        logger.info(f"Prediction successful: {validated_desc} -> {prediction} (confidence: {confidence:.2f})")
        
        return jsonify({
            'predicted_hs_code': str(prediction),
            'predicted_hs_category': category,
            'confidence': round(float(confidence), 2),
            'powered_by': 'DYMRA TECH'
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    if clf is None or bert_model is None:
        logger.error("Model not loaded")
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        
        # Validate file
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "Only CSV files are allowed"}), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size_mb = file.tell() / (1024 * 1024)
        file.seek(0)  # Reset to beginning
        
        if file_size_mb > Config.MAX_FILE_SIZE:
            return jsonify({"error": f"File too large. Maximum size is {Config.MAX_FILE_SIZE}MB"}), 400
        
        df = pd.read_csv(file)
        if 'description' not in df.columns:
            return jsonify({"error": "Missing 'description' column"}), 400
        
        # Validate descriptions
        descriptions = df['description'].fillna('')
        valid_descriptions = []
        invalid_rows = []
        
        for idx, desc in enumerate(descriptions):
            try:
                validated_desc = validate_description(desc)
                valid_descriptions.append(validated_desc)
            except ValueError as e:
                invalid_rows.append(f"Row {idx + 1}: {str(e)}")
        
        if invalid_rows:
            return jsonify({
                "error": "Invalid descriptions found",
                "invalid_rows": invalid_rows
            }), 400
        
        # Process predictions
        desc_embeddings = bert_model.encode(valid_descriptions, show_progress_bar=False)
        probabilities = clf.predict_proba(desc_embeddings)
        df['predicted_hs_code'] = clf.classes_[np.argmax(probabilities, axis=1)]
        df['confidence'] = np.max(probabilities, axis=1).round(2)
        df['predicted_hs_category'] = df['predicted_hs_code'].astype(str).map(hs_code_to_name).fillna("Unknown Category")
        
        # Add metadata to response
        results = df.to_dict(orient='records')
        logger.info(f"Bulk prediction successful: {len(results)} predictions processed")
        
        return jsonify({
            'results': results,
            'total_predictions': len(results),
            'powered_by': 'DYMRA TECH'
        })
    except Exception as e:
        logger.error(f"Bulk prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, port=5001)
