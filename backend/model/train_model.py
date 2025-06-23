import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Load HS code dataset from harmonized-stem.csv
df = pd.read_csv('data/harmonized-stem.csv')
df = df[df['level'] == '2digit']  # Use only 2-digit level rows

X = df['hs_product_name_short_en'].fillna('')
y = df['hs_product_code'].astype(str)

# Create and train the model
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'backend/model/model.pkl')
print("Model trained and saved as model.pkl")
