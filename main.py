from fastapi import FastAPI, HTTPException
from typing import Dict
from pydantic import ValidationError
from settings import ModelParams, ModelClass, LinearParams, ForestParams, TreeParams, KnnParams, ServiceStatus
from mlops_pipeline.models import train_and_save_model, load_model  # import your existing model functions
import os
from fastapi import FastAPI
from typing import Dict
from settings import ModelClass
from pydantic import BaseModel
from typing import List

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

@app.get('/train/')
def train_model(params: ModelParams) -> Dict[str, str]:
    """
    Trains a model based on the given parameters
    """
    try:
        model_class = ModelClass[params.ml_model_type.value]
        model_type = model_class.converters[params.ml_model_type]

        model = model_type(**params.ml_model_params.dict()) 

        X_train, y_train = load_data() 
        model = train_and_save_model(X_train, y_train, config=params)

        model_file = os.path.join(MODEL_DIR, f"{params.ml_model_type.value}_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

        return {"status": "Model trained and saved successfully", "model_type": params.ml_model_type.value}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during training: {e}")

@app.get('/predict/')
def predict(model_type: str, data: Dict[str, float]) -> Dict[str, str]:
    """
    Returns predictions from a trained model.
    """
    model_file = os.path.join(MODEL_DIR, f"{model_type}_model.pkl")
    if os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        input_features = list(data.values())
        prediction = model.predict([input_features])
        
        return {"prediction": prediction.tolist()[0]}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")

@app.get('/remove/')
def remove_trained_model(model_type: str) -> Dict[str, str]:
    """
    Deletes a trained model.
    """
    model_file = os.path.join(MODEL_DIR, f"{model_type}_model.pkl")
    if os.path.exists(model_file):
        os.remove(model_file)
        return {"status": f"Model {model_type} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
