import streamlit as st
import pandas as pd
import pickle
from preprocessing import validate_and_prepare_data, split_train_test  # import your preprocessing functions
from sklearn.metrics import accuracy_score, classification_report
from time import sleep

#Загружаем данные
uploaded_train_file = st.file_uploader("Upload your Titanic training CSV file", type=["csv"], key="train_file")
uploaded_test_file = st.file_uploader("Upload your Titanic test CSV file", type=["csv"], key="test_file")

if uploaded_train_file is not None:
    #Чтобы наблюдать за проихсодящим
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

    #Проверям тест
    if uploaded_test_file is not None:
        with st.spinner('Preprocessing the test data...'):
            #Обрабатываем тестовые данные
            X_test = validate_and_prepare_data(uploaded_test_file, train=False)
            st.write("Test data preprocessing complete!")

        #Даем выбрать модель
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "SVM"])

        if model_choice is not None:
            if model_choice == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=500)
            elif model_choice == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
            elif model_choice == "SVM":
                from sklearn.svm import SVC
                model = SVC()

            #Обновляем прогресс-бар
            progress_bar = st.progress(0)
            progress_text = st.text('Training in progress...')
            
            #Дпропаем 'split' 'target' перед обучением
            X_train_split = X_train_split.drop(columns=['split', 'target'])
            X_val_split = X_val_split.drop(columns=['split', 'target'])

            #Обучение
            model.fit(X_train_split, y_train_split)
            
            progress_bar.progress(100)
            progress_text.text('Training complete!')
            
            #Снимаем метрики
            y_pred_val = model.predict(X_val_split)
            val_accuracy = accuracy_score(y_val_split, y_pred_val)
            st.write(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

            #Репорт
            st.write("Validation Classification Metrics:")
            class_report = classification_report(y_val_split, y_pred_val, output_dict=True)
            class_report_df = pd.DataFrame(class_report).transpose()  # Convert to DataFrame for better readability
            st.table(class_report_df) 

            X_test = X_test.drop(columns=['split', 'target'], errors='ignore') 
            y_pred_test = model.predict(X_test)

            # if uploaded_test_file is not None:
            #     st.write("Predictions on the test data:")
            #     predictions_df = pd.DataFrame(y_pred_test, columns=["Predictions"])
            #     st.write(predictions_df)

            model_path = "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            st.success("Model has been saved!")
            
        else:
            st.warning("Please select a model.")
    else:
        st.warning("Please upload the test CSV file.")
else:
    st.warning("Please upload the training CSV file.")

