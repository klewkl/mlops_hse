import pandas as pd
from fastapi import FastAPI
from typing import Dict
from settings import ModelParams, ModelClass, ServiceStatus, Settings
import logging
from typing import List, Any
from mlops_pipeline.ml_pipeline import parse_data, Model
from pydantic import BaseModel, ValidationError
from io import StringIO
from fastapi import HTTPException
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
import numpy as np
import os
from mlops_pipeline.minio_utils import (
    upload_model_to_minio,
    upload_dataset_to_minio_and_track_with_dvc,
)
from clearml import Task


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
setts = Settings()


class TrainRequestModel(BaseModel):
    ml_model_type: str
    ml_model_params: Dict[str, Any]
    data: str  #
    target_column: str


@app.post("/train/")
async def train_model(request: TrainRequestModel):
    try:
        task = Task.init(
            project_name="MLops Project",
            task_name=f"Train {request.ml_model_type}",
            task_type=Task.TaskTypes.training,
        )

        model_params = ModelParams(
            ml_model_type=request.ml_model_type, ml_model_params=request.ml_model_params
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
            data_df, target=request.target_column, split=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Data parsing and splitting failed: {str(e)}"
        )

    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    local_dataset_path = os.path.join(datasets_dir, "training_data.csv")
    data_df.to_csv(local_dataset_path, index=False)

    bucket_name = "models-bucket"
    dataset_object_name = f"{model_params.ml_model_type}_training_data.csv"
    upload_dataset_to_minio_and_track_with_dvc(
        local_dataset_path, bucket_name, dataset_object_name
    )

    model.train(x_train, y_train)

    local_model_path = model.save("models")

    object_name = f"{model_params.ml_model_type}_model.pkl"
    upload_model_to_minio(local_model_path, bucket_name, object_name)

    task.upload_artifact(name="model", artifact_object=local_model_path)

    task.close()

    project = task.project
    task_url = (
        f"https://app.clear.ml/projects/{project}/experiments/{task.id}/output/log"
    )

    return {
        "status": "success",
        "message": f"Model {model_params.ml_model_type} trained, saved locally, dataset and model uploaded to Minio!",
        "task_id": task.id,
        "task_url": task_url,
    }


class PredictRequestModel(BaseModel):
    model_type: str
    data: str
    target_column: str


@app.post("/predict/")
async def predict(request: PredictRequestModel) -> Dict[str, Any]:
    """
    Returns predictions from a trained model along with accuracy and classification report.
    """
    model = Model()
    model.load(setts.MODEL_DIR, model_type=request.model_type)
    logger.info(f"Model {request.model_type} loaded successfully.")

    task = Task.init(
        project_name="MLops Project",
        task_name=f"Prediction with {request.model_type}",
        task_type=Task.TaskTypes.inference,
        reuse_last_task_id=False,
    )

    task_logger = task.get_logger()

    try:
        task_logger.report_text(f"Model {request.model_type} loaded successfully.")
        task_logger.report_text("Task created successfully.")

        data_df = pd.read_csv(StringIO(request.data))
        task_logger.report_text("Data loaded successfully.")

        x_train, y_train, x_test, y_test = parse_data(
            data_df, target=request.target_column, split=True
        )
        task_logger.report_text(
            f"Data split successfully: x_test shape: {x_test.shape}, y_test shape: {y_test.shape}"
        )

        predictions = model.predict(x_test)

        if isinstance(predictions, np.ndarray):
            predictions = predictions.flatten()

        predictions_list = predictions.tolist()
        accuracy = accuracy_score(y_test, predictions)

        class_report = None
        if len(np.unique(y_test)) == 2:
            class_report = classification_report(y_test, predictions)

        cm = confusion_matrix(y_test, predictions)
        cm_list = cm.tolist()

        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)

        task.upload_artifact(name="confusion_matrix", artifact_object=cm_list)
        task.upload_artifact(name="classification_report", artifact_object=class_report)

        task_logger.report_text(
            f"Prediction completed: accuracy = {accuracy:.4f}, ROC AUC = {roc_auc:.4f}"
        )

        task.close()

        project = task.project
        task_url = (
            f"https://app.clear.ml/projects/{project}/experiments/{task.id}/output/log"
        )

        return {
            "predictions": predictions_list,
            "y_true": y_test.tolist(),
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": cm_list,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "roc_auc": roc_auc,
            "task_id": task.id,
            "task_url": task_url,
        }

    except Exception as e:
        task_logger.report_text(f"Error in prediction: {str(e)}")
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/remove/")
def remove_trained_model(model_type: ModelClass):
    """
    Deletes a trained model.
    """
    model_file = setts.MODEL_DIR / f"{model_type}_model.pkl"
    if model_file.is_file():
        model_file.unlink()


@app.get("/info")
def get_model_list() -> Dict[str, List[str]]:
    """
    Returns a list of models that can be trained
    """
    model_list = [el for el in ModelClass.__members__.values()]
    return {"models_available": model_list}


@app.get("/status")
def get_status() -> Dict[str, ServiceStatus]:
    """
    Returns the current status of the service
    """
    return {"status": ServiceStatus.waiting}
