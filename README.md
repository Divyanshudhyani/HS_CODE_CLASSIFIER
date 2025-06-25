##Overview Project 

The HS Code Classifier is a machine learning-powered tool designed to automatically predict the correct Harmonized System (HS) code for a product based on its textual description. This system streamlines the process of trade and customs classification, reducing manual effort and minimizing errors in assigning HS codes for products.

##Project Structure

hs_code_classifier/
├── backend/
│   ├── app.py                # Flask backend API for predictions
│   └── model/
│       ├── train_model.py    # Model training script
│       └── model.pkl         # Trained model (generated after training)
├── data/
│   ├── harmonized-stem.csv   # Official HS code dictionary (for lookup)
│   └── product_descriptions.csv # Labeled product descriptions for training
├── frontend/
│   └── app.py                # Streamlit frontend UI
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation (this file)



---

## Purpose of Dataset Generation

- **Why a Custom Dataset?**  
  The core of this project is a labeled dataset (`data/product_descriptions.csv`) containing real-world product descriptions and their correct HS codes. This dataset is essential for training a machine learning model that can generalize to new, unseen product descriptions.
- **HS Code Dictionary**  
  The `harmonized-stem.csv` file is used as a lookup table to map predicted HS codes to their official category names, making results more interpretable for users.

---

## Project Strengths

- **End-to-End Solution:**  
  Integrates data preparation, model training, backend API, and a user-friendly frontend.
- **Customizable & Extensible:**  
  Easily improve accuracy by adding more labeled examples to the training dataset.
- **Bulk & Single Prediction:**  
  Supports both individual product lookups and bulk CSV uploads.
- **Confidence Scores:**  
  Provides a confidence score for each prediction, helping users gauge reliability.
- **Clear, Modular Codebase:**  
  Each component (data, model, backend, frontend) is cleanly separated for easy maintenance and upgrades.

---

## How to Implement and Run the Project

### 1. Clone the Repository & Set Up Environment

```bash
git clone <your-repo-url>
cd hs_code_classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare the Dataset

- Edit or expand `data/product_descriptions.csv` with more labeled product descriptions and their correct HS codes.
- The more examples per category, the better the model will perform.

### 3. Train the Model

```bash
python backend/model/train_model.py
```
- This will generate/overwrite `backend/model/model.pkl`.

### 4. Start the Backend API

```bash
python backend/app.py
```
- By default, runs on `http://127.0.0.1:5001` (or change the port if needed).

### 5. Start the Frontend UI

```bash
streamlit run frontend/app.py
```
- Access the UI at `http://localhost:8501` (or the port shown in your terminal).

### 6. Use the Application

- Enter a product description for single prediction, or upload a CSV for bulk predictions.
- The UI will display the predicted HS code, product category, and confidence score.

---

## Conclusion

The HS Code Classifier provides a practical, extensible solution for automating the assignment of HS codes to products using machine learning. By continuously improving the labeled dataset, the system can become increasingly accurate and valuable for businesses and customs professionals. Its modular design makes it easy to maintain, upgrade, and adapt to new requirements.

---

**Tip:**  
For best results, regularly expand your training dataset with new product descriptions and retrain the model!