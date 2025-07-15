import streamlit as st
import requests
import pandas as pd
import os

# Custom CSS for the logo
st.markdown("""
<style>
.logo-container {
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0066cc, #004499);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 30px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.logo-icon {
    width: 60px;
    height: 60px;
    margin-right: 15px;
}

.logo-text {
    color: white;
    font-size: 28px;
    font-weight: bold;
    font-family: 'Arial', sans-serif;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
}

.circuit-node {
    fill: white;
    stroke: white;
    stroke-width: 2;
}

.circuit-line {
    stroke: white;
    stroke-width: 3;
    stroke-linecap: round;
}
</style>
""", unsafe_allow_html=True)

# Logo HTML
logo_html = """
<div class="logo-container">
    <svg class="logo-icon" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg">
        <!-- Circuit board nodes -->
        <circle class="circuit-node" cx="30" cy="15" r="4"/>
        <circle class="circuit-node" cx="20" cy="45" r="4"/>
        <circle class="circuit-node" cx="40" cy="45" r="4"/>
        <circle class="circuit-node" cx="30" cy="8" r="2"/>
        
        <!-- Circuit connections -->
        <line class="circuit-line" x1="30" y1="19" x2="30" y2="35"/>
        <line class="circuit-line" x1="30" y1="35" x2="20" y2="41"/>
        <line class="circuit-line" x1="30" y1="35" x2="40" y2="41"/>
        <line class="circuit-line" x1="30" y1="11" x2="30" y2="6"/>
    </svg>
    <div class="logo-text">DYMRA TECH</div>
</div>
"""

# Display the logo
st.markdown(logo_html, unsafe_allow_html=True)

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
                
                # Show powered by info if available
                if 'powered_by' in data:
                    st.info(f"Powered by {data['powered_by']}")
                
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
            # Handle new API response format
            response_data = res.json()
            
            # Check if response has the new format with 'results' key
            if 'results' in response_data:
                results_data = response_data['results']
                total_predictions = response_data.get('total_predictions', len(results_data))
                st.success(f"✅ Successfully processed {total_predictions} predictions")
                
                # Show powered by info if available
                if 'powered_by' in response_data:
                    st.info(f"Powered by {response_data['powered_by']}")
            else:
                # Fallback for old format
                results_data = response_data
            
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
