import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load your labeled data
print("Loading product descriptions dataset...")
df = pd.read_csv('data/product_descriptions.csv')
X = df['description'].fillna('')
y = df['hs_code'].astype(str)

# Use a pre-trained BERT model for sentence embeddings
bert_model_name = 'all-MiniLM-L6-v2'
print(f"Loading BERT model: {bert_model_name}")
bert_model = SentenceTransformer(bert_model_name)

# Convert product descriptions to BERT embeddings
print("Encoding product descriptions with BERT...")
X_embeddings = bert_model.encode(X.tolist(), show_progress_bar=True)

# Split for evaluation (optional)
X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42)

# Train a classifier on the embeddings
print("Training Logistic Regression classifier on BERT embeddings...")
clf = LogisticRegression(max_iter=2000, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate
print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

# Save the classifier and the BERT model name
joblib.dump({'clf': clf, 'bert_model': bert_model_name}, 'backend/model/model_bert.pkl')
print("✅ BERT-based model saved as model_bert.pkl")
