from minio import Minio
from minio.error import S3Error
import subprocess

minio_client = Minio(
    "minio:9000", access_key="minioadmin", secret_key="minioadmin", secure=False
)


def upload_model_to_minio(local_model_path: str, bucket_name: str, object_name: str):
    """
    Uploads the trained model to Minio storage.
    """
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        minio_client.fput_object(bucket_name, object_name, local_model_path)
        print(f"Model uploaded to Minio: {bucket_name}/{object_name}")
    except S3Error as e:
        print(f"Error uploading model to Minio: {e}")
        raise Exception(f"Error uploading model: {e}")


def upload_dataset_to_minio_and_track_with_dvc(
    local_dataset_path: str, bucket_name: str, object_name: str
):
    """
    Uploads the dataset to Minio and creates a DVC file for versioning.
    """
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)

        minio_client.fput_object(bucket_name, object_name, local_dataset_path)
        print(f"Dataset uploaded to Minio: {bucket_name}/{object_name}")

        dvc_file = f"{local_dataset_path}.dvc"

        subprocess.run(["dvc", "add", local_dataset_path])

        subprocess.run(["git", "commit", "-m", f"Track dataset: {object_name}"])

        subprocess.run(["dvc", "push"])

        print(f"Dataset versioned with DVC: {dvc_file}")

    except S3Error as e:
        print(f"Error uploading dataset to Minio: {e}")
        raise Exception(f"Error uploading dataset: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error in DVC tracking or commit: {e}")
        raise Exception(f"Error in DVC operations: {e}")
