import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


FASTAPI_URL = "http://localhost:8000"  

def fetch_model_list():
    try:
        response = requests.get(f"{FASTAPI_URL}/info")
        if response.status_code == 200:
            model_list = response.json()['models_available']
            return model_list
        else:
            st.error(f"Error fetching model list: {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {str(e)}")
        return []

def get_available_model():
    available_models = fetch_model_list()
    if not available_models:
        st.warning("No models available. Please ensure your FastAPI server is running and accessible.")
    return available_models

def predict_model(model_choice, data_str, target_column):
    try:
        payload = {
            "model_type": model_choice,
            "data": data_str,
            "target_column": target_column
        }
        response = requests.post(f"{FASTAPI_URL}/predict/", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            predictions = np.array(result['predictions'])
            y_true = np.array(result['y_true'])
            accuracy = result['accuracy']
            class_report = result['classification_report']
            cm = np.array(result['confusion_matrix'])
            fpr = np.array(result['fpr'])
            tpr = np.array(result['tpr'])
            roc_auc = result['roc_auc']
            return predictions, y_true, accuracy, class_report, cm, fpr, tpr, roc_auc
        else:
            st.error(f"Error during prediction: {response.text}")
            return None, None, None, None, None, None, None, None
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {str(e)}")
        return None, None, None, None, None, None, None, None


def streamlit_logic():
    uploaded_data = st.file_uploader("Upload your CSV file", type=["csv"], key="data_uploader")

    if uploaded_data is not None:
        data = pd.read_csv(uploaded_data)
        st.write("Data preview:")
        st.dataframe(data.head())

        target_column = st.selectbox("Select the target column", options=data.columns)
        model_choice = st.selectbox("Select Model", get_available_model())

        if model_choice is not None:
            model_params = {
                "ml_model_type": model_choice,
                "ml_model_params": {}, 
            }

            data_str = uploaded_data.getvalue().decode("utf-8")
            train_button = st.button("Train Model")
            if train_button:
                payload = {
                    "ml_model_type": model_choice,
                    "ml_model_params": model_params,
                    "data": data_str,  
                    "target_column": target_column  
                }

                with st.spinner(f'Training the {model_choice} model...'):
                    try:
                        response = requests.post(f"{FASTAPI_URL}/train/", json=payload)
                        if response.status_code == 200:
                            st.success(f"Model {model_choice} trained and saved successfully!")
                        else:
                            st.error(f"Error training model: {response.text}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Request Error: {str(e)}")
            
            if st.button("Make Prediction"):
                with st.spinner(f'Making predictions with {model_choice}...'):
                    predictions, y_true, accuracy, class_report, cm, fpr, tpr, roc_auc = predict_model(model_choice, data_str, target_column)
                    
                    if predictions is not None:

                        # st.write("Predictions:")
                        # st.write(predictions)

                        if target_column in data.columns:
                            # st.write(f"Accuracy: {accuracy:.4f}")
                            # st.write("Classification Report:")
                            # st.text(class_report)

                            #CF
                            st.subheader("Confusion Matrix")
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                            plt.xlabel('Predicted')
                            plt.ylabel('True')
                            st.pyplot(fig)

                            #ROC-AUC
                            st.subheader("ROC Curve")
                            fig_roc, ax_roc = plt.subplots()
                            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            ax_roc.set_xlim([0.0, 1.0])
                            ax_roc.set_ylim([0.0, 1.05])
                            ax_roc.set_xlabel('False Positive Rate')
                            ax_roc.set_ylabel('True Positive Rate')
                            ax_roc.set_title('Receiver Operating Characteristic (ROC)')
                            ax_roc.legend(loc="lower right")
                            st.pyplot(fig_roc)
                            
                            st.subheader("Classification Report")
                            st.text(class_report)
                            
    else:
        st.warning("Please upload the CSV file.")

streamlit_logic()