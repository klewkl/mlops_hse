import pandas as pd
from fastapi import FastAPI
from typing import Dict
from settings import ModelParams, ModelClass, ServiceStatus, Settings
import logging
from typing import List, Any, Union
from mlops_pipeline.ml_pipeline import parse_data, Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
setts = Settings()


@app.get('/info')
def get_model_list() -> Dict[str, List[str]]:
    """
    Returns a list of models that can be trained
    """
    model_list = [el for el in ModelClass.__members__.values()]
    return {'models_available': model_list}


@app.get('/status')
def get_status() -> Dict[str, ServiceStatus]:
    """
    Returns the current status of the service
    """
    return {"status": ServiceStatus.waiting}


@app.post('/train/')
async def train_model(params: ModelParams, data: str):
    """
    Trains a model based on the given parameters.
    """
    model = Model(params)
    logger.info(f"Selected model type: {model.model_type}")
    logger.info("Model initialized successfully.")

    x_train, y_train, x_test, y_test = parse_data(data)
    logger.info("Training data loaded.")

    logger.info(f"Training the model with params: {params.ml_model_params}...")
    model = model.train(x_train, y_train)
    logger.info("Model trained successfully.")

    model.save(setts.MODEL_DIR)
    logger.info("Model saved successfully.")


@app.post('/predict/')
async def predict(model_type: str, data: str) -> Dict[str, Any]:
    """
    Returns predictions from a trained model.
    """
    model = Model()
    model.load(setts.MODEL_DIR)
    logger.info(f"Model loaded successfully: {model_type}")

    data = parse_data(data, train=False)
    return {'predictions': model.predict(data)}


@app.get('/remove/')
def remove_trained_model(model_type: ModelClass):
    """
    Deletes a trained model.
    """
    model_file = setts.MODEL_DIR / f'{model_type}_model.pkl'
    if model_file.is_file():
        model_file.unlink()
