import pandas as pd
from fastapi import FastAPI
from typing import Dict
from settings import ModelParams, ModelClass, ServiceStatus, Settings
import logging
from typing import List, Any, Union
from mlops_pipeline.ml_pipeline import parse_data, Model
from pydantic import BaseModel
from io import StringIO
from fastapi import FastAPI, HTTPException
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import numpy as np
import pandas as pd
import json
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
setts = Settings()


class TrainRequestModel(BaseModel):
    ml_model_type: str  
    ml_model_params: Dict[str, Any]  
    data: str  # 
    target_column: str  

@app.post('/train/')
async def train_model(request: TrainRequestModel):
    try:
        model_params = ModelParams(
            ml_model_type=request.ml_model_type, 
            ml_model_params=request.ml_model_params
        )
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")

    model = Model(model_params)  
    
    try:
        data_df = pd.read_csv(StringIO(request.data))  
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    try:
        x_train, y_train, x_test, y_test = parse_data(
            data_df, 
            target=request.target_column,  
            split=True  
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data parsing and splitting failed: {str(e)}")

    model.train(x_train, y_train)
    model.save('models') 

    return {"status": "success", "message": f"Model {model_params.ml_model_type} trained and saved successfully!"}

class PredictRequestModel(BaseModel):
    model_type: str  
    data: str  
    target_column: str  

@app.post('/predict/')
async def predict(request: PredictRequestModel) -> Dict[str, Any]:
    """
    Returns predictions from a trained model along with accuracy and classification report.
    """
    model = Model() 
    model.load(setts.MODEL_DIR, model_type=request.model_type)  
    logger.info(f"Model {request.model_type} loaded successfully.")

    try:

        data_df = pd.read_csv(StringIO(request.data))  
        logger.info("Data loaded successfully.")

        x_train, y_train, x_test, y_test = parse_data(data_df, target=request.target_column, split=True)
        logger.info(f"Data split successfully: x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

        predictions = model.predict(x_test)

        if isinstance(predictions, np.ndarray):
            predictions = predictions.flatten()

        predictions_list = predictions.tolist()
        accuracy = accuracy_score(y_test, predictions)
        # logger.info(f"Accuracy on test set: {accuracy:.4f}")

        class_report = None
        if len(np.unique(y_test)) == 2:  
            class_report = classification_report(y_test, predictions)
            # logger.info(f"Classification Report:\n{class_report}")

        #CF
        cm = confusion_matrix(y_test, predictions)
        cm_list = cm.tolist() 

        # ROC-AUC
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)

        return {
            'predictions': predictions_list,
            'y_true': y_test.tolist(),  
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm_list,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'roc_auc': roc_auc
        }

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get('/remove/')
def remove_trained_model(model_type: ModelClass):
    """
    Deletes a trained model.
    """
    model_file = setts.MODEL_DIR / f'{model_type}_model.pkl'
    if model_file.is_file():
        model_file.unlink()

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