from typing import Optional, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split

model_dict = {
    'logistic_regression': LogisticRegression,
    'svm': LinearSVC,
    'random_forest': RandomForestClassifier,
    'knn': KNeighborsClassifier,
    'decision_tree': DecisionTreeClassifier,
}

default_params_dict = {
    'logistic_regression': {'C': 1, 'max_iter': 100},
    'svm': {'C': 1, 'kernel': 'rbf'},
    'random_forest': {'n_estimators': 100, 'max_depth': None},
    'knn': {'n_neighbors': 5, 'algorithm': 'auto'},
    'decision_tree': {'max_depth': None},
}

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def create_model(model_type: str, model_params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Creates a model pipeline.

    :param model_type: Model type (logistic_regression, svm, etc.)
    :param model_params: Hyperparameters for the model
    :return: Model pipeline (sklearn)
    """
    assert model_type in model_dict, f"Model {model_type} is not supported."

    if model_params is None:
        model_params = {}

    model = model_dict[model_type](**model_params)

    return model


def train_and_save_model(X: pd.DataFrame, y: pd.Series, model_type: str, model_params: Optional[Dict[str, Any]] = None):
    """
    Trains and saves the model.

    :param X: Feature matrix for training
    :param y: Target vector for training
    :param model_type: Type of model (logistic_regression, svm, etc.)
    :param model_params: Hyperparameters for the model
    :return: Trained model
    """

    model = create_model(model_type, model_params)

    model.fit(X, y)

    model_file = os.path.join(MODEL_DIR, f"{model_type}_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model trained and saved as {model_file}")
    return model


def load_model(model_type: str):
    """
    Loads a trained model from the saved file.

    :param model_type: Type of model to load (logistic_regression, svm, etc.)
    :return: Loaded model
    """
    model_file = os.path.join(MODEL_DIR, f"{model_type}_model.pkl")
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"Model {model_type} loaded successfully.")
        return model
    else:
        print(f"Model {model_type} does not exist.")
        return None