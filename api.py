import datetime
from typing import Optional, Dict, Union
from fastapi import FastAPI, HTTPException
from settings import ServiceStatus, Settings, ModelParams
from mlops_pipeline.models import train_and_save_model
import pandas as pd


app = FastAPI()

config = Settings()
STATUS = ServiceStatus.waiting

feature_columns = ['Age', 'Name', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
target_column = 'Survived'


@app.get('/status')
def get_status() -> Dict[str, ServiceStatus]:
    """
    Function to get statis of api
    :return: dict containing status
    """
    return {
        "status": STATUS,
    }


@app.get('/info')
def get_model_list() -> Dict[str, str]:
    """
    Function to get info by api
    :return: dict containing models list
    """
    return {
        "models": ...
    }


# @app.get('/train/')
# def train(params: ModelParams) -> Dict[str, str]:
#     df = pd.read_csv('data/test_data.csv')
#     train_data = df[df['split'] == 'train']
#     train_and_save_model(train_data[feature_columns], train_data[target_column], config=params)
#     return {'key': 'key'}


@app.get('/predict/')
def predict() -> Dict[str, str]:
    pass


@app.get('/remove/')
def remove() -> Dict[str, str]:
    pass


@app.get('/list/')
def list_trained_models() -> Dict[str, str]:
    pass
