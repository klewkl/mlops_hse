import streamlit as st
import pandas as pd
import pickle
import requests
from preprocessing import validate_and_prepare_data, split_train_test
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

FASTAPI_URL = "http://127.0.0.1:8000"  

def fetch_model_list():
    try:
        response = requests.get(f"{FASTAPI_URL}/info")
        if response.status_code == 200:
            model_list = response.json()['models']
            return model_list
        else:
            st.error(f"Error fetching model list: {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {str(e)}")
        return []

available_models = fetch_model_list()

if not available_models:
    st.warning("No models available. Please ensure your FastAPI server is running and accessible.")

uploaded_train_file = st.file_uploader("Upload your Titanic training CSV file", type=["csv"], key="train_file")
uploaded_test_file = st.file_uploader("Upload your Titanic test CSV file", type=["csv"], key="test_file")

if uploaded_train_file is not None:
    with st.spinner('Preprocessing the training data...'):
        X_train, y_train = validate_and_prepare_data(uploaded_train_file, train=True)
        X_train_split, X_val_split, y_train_split, y_val_split = split_train_test(X_train, y_train)

        X_train_split['target'] = y_train_split
        X_train_split['split'] = 'train'
        X_val_split['target'] = y_val_split
        X_val_split['split'] = 'test'

        df = pd.concat([X_train_split, X_val_split], ignore_index=True)
        df.to_csv('preprocessed_data.csv', index=False) 
        st.write("Training data preprocessing complete!")

    if uploaded_test_file is not None:
        with st.spinner('Preprocessing the test data...'):
            X_test = validate_and_prepare_data(uploaded_test_file, train=False)
            st.write("Test data preprocessing complete!")


        model_choice = st.selectbox("Select Model", available_models)

        if model_choice is not None:

            model_params = {
                "ml_model_type": model_choice.lower().replace(" ", "_"),  
                "ml_model_params": {} 
            }

            with st.spinner(f'Training the {model_choice} model...'):
                try:
                    response = requests.post(f"{FASTAPI_URL}/train/", json=model_params)
                    
                    if response.status_code == 200:
                        st.success(f"Model {model_choice} trained and saved successfully!")
                    else:
                        st.error(f"Error training model: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request Error: {str(e)}")

            if uploaded_test_file is not None:
                X_test = X_test.drop(columns=['split', 'target'], errors='ignore')  # Ensure columns are aligned

                prediction_params = {
                    "model_type": model_choice.lower().replace(" ", "_"),  # Ensure correct model type formatting
                    "data": X_val_split.drop(columns=['split', 'target'], errors='ignore').to_dict(orient="records")  
                }

                with st.spinner(f'Making predictions with the {model_choice} model...'):
                    try:
                        prediction_response = requests.post(f"{FASTAPI_URL}/predict/", json=prediction_params)

                        if prediction_response.status_code == 200:
                            y_pred_val = prediction_response.json()['prediction']
                            
                            if isinstance(y_pred_val, list):
                                st.write(f"Validation Accuracy: {accuracy_score(y_val_split, y_pred_val) * 100:.2f}%")

                                st.write("Validation Classification Metrics:")
                                class_report = classification_report(y_val_split, y_pred_val, output_dict=True)
                                class_report_df = pd.DataFrame(class_report).transpose() 
                                st.table(class_report_df)
                            else:
                                st.error("Prediction response is not in expected format. Please check the backend.")
                        else:
                            st.error(f"Error making prediction: {prediction_response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Request Error: {str(e)}")

        else:
            st.warning("Please select a model.")
    else:
        st.warning("Please upload the test CSV file.")
else:
    st.warning("Please upload the training CSV file.")
