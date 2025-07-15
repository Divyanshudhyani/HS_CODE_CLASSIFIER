# HS Code Classifier

## Overview

The HS Code Classifier is a machine learning-powered tool designed to automatically predict the correct Harmonized System (HS) code for a product based on its textual description. This system uses BERT (Bidirectional Encoder Representations from Transformers) embeddings combined with Logistic Regression to provide accurate HS code predictions, streamlining the process of trade and customs classification.

## Project Structure

```
hs_code_classifier/
├── backend/
│   ├── app.py                    # Flask backend API for predictions
│   └── model/
│       ├── train_model.py        # Model training script using BERT
│       ├── model.pkl             # Legacy model (2.5MB)
│       └── model_bert.pkl        # BERT-based model (164KB)
├── data/
│   ├── harmonized-stem.csv       # Official HS code dictionary (6407 entries)
│   └── product_descriptions.csv  # Labeled product descriptions (912 entries)
├── frontend/
│   └── app.py                    # Streamlit frontend UI
├── generated_product_descriptions.csv  # Additional generated data
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Technology Stack

- **Backend**: Flask API with CORS support
- **Frontend**: Streamlit web interface
- **Machine Learning**: 
  - BERT embeddings using `sentence-transformers` (all-MiniLM-L6-v2)
  - Logistic Regression classifier with balanced class weights
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib

## Key Features

- **BERT-Powered Predictions**: Uses state-of-the-art BERT embeddings for better understanding of product descriptions
- **Single & Bulk Predictions**: Support for individual product lookups and CSV batch processing
- **Confidence Scoring**: Provides confidence scores to help users gauge prediction reliability
- **Rule-based Fallbacks**: Includes fallback rules for low-confidence predictions
- **Real-time API**: RESTful API endpoints for integration with other systems
- **User-friendly Interface**: Clean Streamlit UI with helpful warnings and tips

## Dataset Information

### Training Data (`data/product_descriptions.csv`)
- **912 labeled examples** covering diverse product categories
- Includes electronics, clothing, furniture, food items, and more
- Each entry contains a product description and its corresponding HS code

### HS Code Dictionary (`data/harmonized-stem.csv`)
- **6,407 official HS codes** with category names
- Used for mapping predicted codes to human-readable categories
- International standard for trade classification

## Installation & Setup

### 1. Clone and Set Up Environment

```bash
git clone <your-repo-url>
cd hs_code_classifier
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Additional Dependencies

The project uses BERT models which require additional packages:

```bash
pip install sentence-transformers torch
```

### 3. Train the Model

```bash
cd backend/model
python train_model.py
```

This will:
- Load the training dataset
- Generate BERT embeddings for all product descriptions
- Train a Logistic Regression classifier
- Save the model as `model_bert.pkl`
- Display training and test accuracy

### 4. Start the Backend API

```bash
cd backend
python app.py
```

The API will run on `http://127.0.0.1:5001` by default.

### 5. Start the Frontend UI

```bash
cd frontend
streamlit run app.py
```

Access the UI at `http://localhost:8501`

## API Endpoints

### Single Prediction
- **POST** `/predict`
- **Body**: `{"description": "product description"}`
- **Response**: 
  ```json
  {
    "predicted_hs_code": "85",
    "predicted_hs_category": "Electrical machinery and equipment",
    "confidence": 0.92
  }
  ```

### Bulk Prediction
- **POST** `/bulk_predict`
- **Body**: CSV file with 'description' column
- **Response**: Array of prediction objects with original descriptions and predictions

## Usage Examples

### Single Product Prediction
1. Enter a detailed product description (e.g., "iPhone 15 Pro Max smartphone 256GB")
2. Click "Predict"
3. View the predicted HS code, category, and confidence score

### Bulk Processing
1. Prepare a CSV file with a 'description' column
2. Upload the file through the Streamlit interface
3. Download results with predictions for all products

## Model Performance

The BERT-based model provides:
- **Better semantic understanding** of product descriptions
- **Improved accuracy** compared to traditional TF-IDF approaches
- **Robust handling** of variations in product descriptions
- **Confidence scoring** for prediction reliability

## Customization & Extension

### Adding Training Data
1. Edit `data/product_descriptions.csv`
2. Add new product descriptions with correct HS codes
3. Retrain the model using `train_model.py`

### Improving Accuracy
- Add more examples for underrepresented HS codes
- Include regional product variations
- Use more specific product descriptions
- Consider domain-specific BERT models for specialized industries

### API Integration
The Flask backend can be easily integrated with:
- E-commerce platforms
- Customs management systems
- Trade compliance tools
- Inventory management systems

## Troubleshooting

### Common Issues
1. **Model loading errors**: Ensure `model_bert.pkl` exists in `backend/model/`
2. **Connection errors**: Verify backend is running on port 5001
3. **Low confidence predictions**: Provide more detailed product descriptions
4. **Memory issues**: BERT models require sufficient RAM (4GB+ recommended)

### Performance Tips
- Use specific product descriptions (e.g., "plastic garden chair" vs "chair")
- Include brand names and specifications when available
- For bulk processing, consider processing in smaller batches

## Dependencies

```
flask
flask-cors
joblib
pandas
scikit-learn
streamlit
sentence-transformers
torch
numpy
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

---

**Note**: For optimal results, regularly update the training dataset with new product descriptions and retrain the model. The more diverse and comprehensive your training data, the better the model will perform on real-world product descriptions.