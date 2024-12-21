# import grpc
# import model_pb2
# import model_pb2_grpc
# import pandas as pd
# from io import StringIO
# import numpy as np

# def train_model(stub, model_type, model_params, data_str, target_column):
#     request = model_pb2.TrainRequest(
#         model_type=model_type,
#         ml_model_params=model_params,
#         data=data_str,
#         target_column=target_column
#     )

#     response = stub.TrainModel(request)
#     print(f"Train Response: {response.status} - {response.message}")

# def predict_model(stub, model_type, data_str, target_column):
#     request = model_pb2.PredictRequest(
#         model_type=model_type,
#         data=data_str,
#         target_column=target_column
#     )

#     response = stub.PredictModel(request)
#     print("Prediction Response:")
#     print(f"Accuracy: {response.accuracy}")
#     print(f"Predictions: {response.predictions}")
#     print(f"Confusion Matrix: {response.confusion_matrix}")
#     print(f"ROC AUC: {response.roc_auc}")

# def run():
#     channel = grpc.insecure_channel('localhost:50051')
#     stub = model_pb2_grpc.ModelServiceStub(channel)

#     # Simulate CSV data
#     uploaded_data = """
#     feature1,feature2,target
#     1.0,2.0,3.0
#     3.0,4.0,7.0
#     5.0,6.0,11.0
#     """

#     data = pd.read_csv(StringIO(uploaded_data))
#     print("Data preview:")
#     print(data.head())

#     target_column = "target"
#     model_choice = "random_forest"  # Example model choice

#     model_params = {
#         "n_estimators": "100",
#         "max_depth": "10",
#         "min_samples_split": "2",
#         "min_samples_leaf": "1"
#     }

#     # Train the model
#     train_model(stub, model_choice, model_params, uploaded_data, target_column)

#     # Make predictions using the trained model
#     predict_model(stub, model_choice, uploaded_data, target_column)

# if __name__ == '__main__':
#     run()

import grpc
import model_pb2
import model_pb2_grpc
import pandas as pd
from io import StringIO


def train_model(stub, model_type, model_params, data_str, target_column):
    # Convert model_params to a map<string, string> format
    model_params_map = {key: value for key, value in model_params.items()}

    # Create TrainRequest
    request = model_pb2.TrainRequest(
        model_type=model_type,
        ml_model_params=model_params_map,  # Ensure it's passed as a map
        data=data_str,
        target_column=target_column,
    )

    try:
        # Call TrainModel RPC
        response = stub.TrainModel(request)
        print(f"Train Response: {response.status} - {response.message}")
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()} - {e.details()}")


def predict_model(stub, model_type, data_str, target_column):
    request = model_pb2.PredictRequest(
        model_type=model_type, data=data_str, target_column=target_column
    )

    try:
        # Call PredictModel RPC
        response = stub.PredictModel(request)
        print("Prediction Response:")
        print(f"Accuracy: {response.accuracy}")
        print(f"Predictions: {response.predictions}")
        print(f"Confusion Matrix: {response.confusion_matrix}")
        print(f"ROC AUC: {response.roc_auc}")
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()} - {e.details()}")


def run():
    # Create gRPC channel and stub
    channel = grpc.insecure_channel("localhost:50051")
    stub = model_pb2_grpc.ModelServiceStub(channel)

    # Simulate CSV data
    uploaded_data = """
    feature1,feature2,target
    1.0,2.0,3.0
    3.0,4.0,7.0
    5.0,6.0,11.0
    """

    data = pd.read_csv(StringIO(uploaded_data))
    print("Data preview:")
    print(data.head())

    target_column = "target"
    model_choice = "random_forest"  # Example model choice

    # Example model parameters
    model_params = {
        "n_estimators": "100",
        "max_depth": "10",
        "min_samples_split": "2",
        "min_samples_leaf": "1",
    }

    # Train the model
    train_model(stub, model_choice, model_params, uploaded_data, target_column)

    # Make predictions using the trained model
    predict_model(stub, model_choice, uploaded_data, target_column)


if __name__ == "__main__":
    run()
