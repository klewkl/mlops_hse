import grpc
from concurrent import futures
import model_pb2
import model_pb2_grpc
import pickle
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from mlops_pipeline.ml_pipeline import Model, parse_data  #
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService(model_pb2_grpc.ModelServiceServicer):

    def TrainModel(self, request, context):
        try:
            model_params = request.ml_model_params
            model_type = request.model_type
            data_str = request.data
            target_column = request.target_column

            data_df = pd.read_csv(StringIO(data_str))
            x_train, y_train, x_test, y_test = parse_data(data_df, target=target_column, split=True)

            model = Model(model_params=model_params, model_type=model_type)

            model.train(x_train, y_train)
            model.save('models') 

            return model_pb2.TrainResponse(status="success", message=f"Model {model_type} trained and saved successfully!")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return model_pb2.TrainResponse(status="failure", message=f"Error: {str(e)}")

    def PredictModel(self, request, context):
        try:
            model_type = request.model_type
            data_str = request.data
            target_column = request.target_column

            model = Model()
            model.load('models', model_type)

            data_df = pd.read_csv(StringIO(data_str))

            x_train, y_train, x_test, y_test = parse_data(data_df, target=target_column, split=True)

            predictions = model.predict(x_test)

            if isinstance(predictions, np.ndarray):
                predictions = predictions.flatten()

            accuracy = accuracy_score(y_test, predictions)

            class_report = None
            if len(np.unique(y_test)) == 2: 
                class_report = classification_report(y_test, predictions)

            cm = confusion_matrix(y_test, predictions)
            cm_list = cm.tolist()

            # ROC-AUC
            fpr, tpr, thresholds = roc_curve(y_test, predictions)
            roc_auc = roc_auc_score(y_test, predictions)

            return model_pb2.PredictResponse(
                predictions=predictions.tolist(),
                y_true=y_test.tolist(),
                accuracy=accuracy,
                classification_report=class_report if class_report else "",
                confusion_matrix=cm_list,
                fpr=fpr.tolist(),
                tpr=tpr.tolist(),
                roc_auc=roc_auc
            )
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return model_pb2.PredictResponse(
                predictions=[],
                y_true=[],
                accuracy=0.0,
                classification_report="",
                confusion_matrix=[],
                fpr=[],
                tpr=[],
                roc_auc=0.0
            )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_ModelServiceServicer_to_server(ModelService(), server)
    server.add_insecure_port('[::]:50051')  # gRPC listens on port 50051
    server.start()
    print("Server started at port 50051.")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

