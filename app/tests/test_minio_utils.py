#PYTHONPATH=app pytest app/tests/test_minio_utils.py

import pytest
from unittest import mock
from minio.error import S3Error
from mlops_pipeline.minio_utils import upload_model_to_minio, upload_dataset_to_minio_and_track_with_dvc
import subprocess


@pytest.fixture
def mock_minio_client():
    """
    Фикстура для создания мока клиента Minio. 

    Эта фикстура используется для замещения реального клиента Minio в тестах.
    Используется для проверки взаимодействия с объектами Minio в тестах.

    Возвращает:
        mock.Mock: Замок клиента Minio.
    """
    with mock.patch('mlops_pipeline.minio_utils.minio_client') as mock_client:
        yield mock_client



def test_upload_model_to_minio_success(mock_minio_client):
    """
    Тест успешной загрузки модели в Minio.

    Этот тест проверяет, что модель успешно загружается в Minio, если 
    ведро уже существует. Ожидается, что метод `fput_object` будет вызван
    с правильными параметрами.

    Аргументы:
        mock_minio_client (mock.Mock): Замок клиента Minio.
    """
    mock_minio_client.bucket_exists.return_value = True  
    mock_minio_client.fput_object.return_value = None  

    local_model_path = 'path/to/local/model.pkl'
    bucket_name = 'models-bucket'
    object_name = 'model.pkl'

    upload_model_to_minio(local_model_path, bucket_name, object_name)

    mock_minio_client.fput_object.assert_called_once_with(bucket_name, object_name, local_model_path)


def test_upload_model_to_minio_bucket_creation(mock_minio_client):
    """
    Тест загрузки модели в Minio с созданием нового ведра.

    Этот тест проверяет сценарий, когда ведро не существует, и в этом случае
    создается новое ведро перед загрузкой модели.

    Аргументы:
        mock_minio_client (mock.Mock): Замок клиента Minio.
    """
    mock_minio_client.bucket_exists.return_value = False  
    mock_minio_client.make_bucket.return_value = None  
    mock_minio_client.fput_object.return_value = None  

    local_model_path = 'path/to/local/model.pkl'
    bucket_name = 'new-model-bucket'
    object_name = 'model.pkl'

    upload_model_to_minio(local_model_path, bucket_name, object_name)

    mock_minio_client.make_bucket.assert_called_once_with(bucket_name)
    mock_minio_client.fput_object.assert_called_once_with(bucket_name, object_name, local_model_path)


# Тест для upload_dataset_to_minio_and_track_with_dvc
def test_upload_dataset_to_minio_and_track_with_dvc(mock_minio_client):
    """
    Тест успешной загрузки набора данных в Minio и отслеживания с помощью DVC.

    Этот тест проверяет, что набор данных успешно загружается в Minio, а также
    выполняются команды DVC для отслеживания и коммита данных.

    Аргументы:
        mock_minio_client (mock.Mock): Замок клиента Minio.
    """
    mock_minio_client.bucket_exists.return_value = True
    mock_minio_client.fput_object.return_value = None  

    with mock.patch('subprocess.run') as mock_run:
        mock_run.return_value = None  

        local_dataset_path = 'path/to/local/dataset.csv'
        bucket_name = 'datasets-bucket'
        object_name = 'dataset.csv'

        upload_dataset_to_minio_and_track_with_dvc(local_dataset_path, bucket_name, object_name)

        mock_minio_client.fput_object.assert_called_once_with(bucket_name, object_name, local_dataset_path)
        mock_run.assert_any_call(["dvc", "add", local_dataset_path])
        mock_run.assert_any_call(["git", "commit", "-m", f"Track dataset: {object_name}"])
        mock_run.assert_any_call(["dvc", "push"])


def test_upload_dataset_to_minio_and_track_with_dvc_failure(mock_minio_client):
    """
    Тест неудачной загрузки набора данных в Minio с отслеживанием через DVC.

    Этот тест проверяет сценарий, когда при выполнении команд DVC возникает ошибка.
    Ожидается, что будет выброшено исключение с соответствующим сообщением об ошибке.

    Аргументы:
        mock_minio_client (mock.Mock): Замок клиента Minio.
    """
    mock_minio_client.bucket_exists.return_value = True
    mock_minio_client.fput_object.return_value = None 

    with mock.patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, 'dvc') 

        local_dataset_path = 'path/to/local/dataset.csv'
        bucket_name = 'datasets-bucket'
        object_name = 'dataset.csv'

        try:
            upload_dataset_to_minio_and_track_with_dvc(local_dataset_path, bucket_name, object_name)
            result = False
        except Exception as e:
            result = str(e)  

        assert result == "Error in DVC operations: Command 'dvc' returned non-zero exit status 1."
