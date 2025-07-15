import streamlit as st
import requests
import pandas as pd
import os

st.title("HS Code Classifier")

# Use environment variables for backend URL for flexibility
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5001")

# Single input prediction
desc = st.text_input("Enter product description")
if st.button("Predict"):
    if not desc.strip():
        st.warning("Please enter a product description.")
    else:
        try:
            res = requests.post(f"{BACKEND_URL}/predict", json={"description": desc})
            if res.ok:
                data = res.json()
                st.success(f"Predicted HS Code: {data['predicted_hs_code']}")
                st.info(f"Product Category: {data['predicted_hs_category']}")
                st.info(f"Confidence: {data['confidence']:.2f}")
                # Warn if input is too generic
                if len(desc.strip().split()) < 2:
                    st.warning("Please enter a more detailed product description for better results (e.g., 'plastic toy car', 'plastic water bottle').")
                # Warn if confidence is low
                if 'confidence' in data and data['confidence'] < 0.3:
                    st.warning("The model is not confident in this prediction. Please provide a more specific product description for higher accuracy.")
            else:
                st.error(f"Prediction failed: {res.json().get('error', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Please ensure the backend is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# Bulk upload prediction
st.write("### Bulk CSV Upload")
uploaded_file = st.file_uploader("Upload a CSV file with a 'description' column", type=['csv'])
if uploaded_file:
    try:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
        res = requests.post(f"{BACKEND_URL}/bulk_predict", files=files)
        
        if res.ok:
            # The backend now returns a list of JSON objects
            results_data = res.json()
            df = pd.DataFrame(results_data)
            
            # Reorder columns for better presentation
            if not df.empty:
                cols_order = ['description', 'predicted_hs_code', 'predicted_hs_category', 'confidence']
                df = df[[col for col in cols_order if col in df.columns]]

            st.dataframe(df)
        else:
            st.error(f"Bulk prediction failed: {res.json().get('error', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the backend. Please ensure the backend is running.")
    except Exception as e:
        st.error(f"An unexpected error occurred during bulk upload: {e}")

# Add a tip for best results
st.info("Tip: For best results, enter a full product description (e.g., 'plastic garden chair', 'leather wallet brown', 'LED television 55 inch').")
