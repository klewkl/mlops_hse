from fastapi import FastAPI, HTTPException
from typing import Dict
from pydantic import ValidationError
from settings import ModelParams, ModelClass, ServiceStatus
from mlops_pipeline.models import train_and_save_model, load_model
import os
import logging
import pickle
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class ModelListResponse(BaseModel):
    models: List[str]

@app.get('/info', response_model=ModelListResponse)
def get_model_list() -> ModelListResponse:
    """
    Returns a list of models that can be trained
    """
    try:
        model_list = [model.value for model in ModelClass]
        return ModelListResponse(models=model_list)
    except Exception as e:
        return {"error": f"Failed to retrieve model list: {str(e)}"}

@app.get('/status')
def get_status() -> Dict[str, ServiceStatus]:
    """
    Returns the current status of the service
    """
    return {"status": ServiceStatus.waiting}

@app.post('/train/')
async def train_model(params: ModelParams) -> Dict[str, str]:
    """
    Trains a model based on the given parameters.
    """
    try:
        logger.info(f"Starting model training with params: {params}")
        
        model_class = ModelClass[params.ml_model_type.value]
        model_type = model_class.converters[params.ml_model_type]
        logger.info(f"Selected model type: {model_type}")

        model = model_type(**params.ml_model_params.dict()) 
        logger.info("Model initialized successfully.")

        X_train, y_train = load_data()
        logger.info("Training data loaded.")

        logger.info("Training the model...")
        model = train_and_save_model(X_train, y_train, config=params)
        logger.info("Model trained successfully.")

        model_file = os.path.join("models", f"{params.ml_model_type.value}_model.pkl")
        logger.info(f"Saving model to: {model_file}")
        
        if not os.path.exists("models"):
            os.makedirs("models")
            logger.info(f"Created model directory: models")

        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info("Model saved successfully.")
        
        return {"status": "Model trained and saved successfully", "model_type": params.ml_model_type.value}
    
    except ValidationError as e:
        logger.error(f"Validation Error: {e}")
        raise HTTPException(status_code=400, detail=f"Validation Error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise HTTPException(status_code=500, detail=f"Error during training: {str(e)}")

@app.post('/predict/')
async def predict(model_type: str, data: Dict[str, float]) -> Dict[str, str]:
    """
    Returns predictions from a trained model.
    """
    try:
        logger.info(f"Received prediction request for model {model_type} with data: {data}")
        
        model_file = os.path.join("models", f"{model_type}_model.pkl")
        logger.info(f"Checking if model exists at: {model_file}")
        
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded successfully: {model_type}")

            input_features = list(data.values())
            logger.info(f"Input features: {input_features}")

            prediction = model.predict([input_features])
            logger.info(f"Prediction result: {prediction}")

            return {"prediction": prediction.tolist()[0]}
        
        else:
            logger.error(f"Model {model_type} not found.")
            raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get('/remove/')
def remove_trained_model(model_type: str) -> Dict[str, str]:
    """
    Deletes a trained model.
    """
    model_file = os.path.join("models", f"{model_type}_model.pkl")
    if os.path.exists(model_file):
        os.remove(model_file)
        return {"status": f"Model {model_type} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")