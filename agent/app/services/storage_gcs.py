from google.cloud import storage
from pathlib import Path
import os

def download_from_gcs(bucket_name: str, remote_path: str, local_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    Path(local_path).mkdir(parents=True, exist_ok=True)

    print(f"⬇Downloading {remote_path} from bucket {bucket_name}...")
    for blob in bucket.list_blobs(prefix=remote_path):
        if blob.name.endswith("/"):
            continue
        dest = Path(local_path) / Path(blob.name).name
        blob.download_to_filename(dest)
    print("Download complete.")


def upload_to_gcs(bucket_name: str, local_path: str, remote_path: str):
    """Upload local_path/* to gs://bucket_name/remote_path"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    print(f"⬆Uploading {local_path} → gs://{bucket_name}/{remote_path}/ ...")
    for file in Path(local_path).rglob("*"):
        if file.is_file():
            blob = bucket.blob(f"{remote_path}/{file.name}")
            blob.upload_from_filename(str(file))
    print("Upload complete.")


def ensure_local_data():
    """Run on startup"""
    bucket = os.getenv("GCS_BUCKET", "ai-mentor-data")
    download_from_gcs(bucket, "chroma_db", "chroma_db")
    download_from_gcs(bucket, "data", "data")


def sync_back():
    """Call before shutdown or periodically to persist updates"""
    bucket = os.getenv("GCS_BUCKET", "ai-mentor-data")
    upload_to_gcs(bucket, "chroma_db", "chroma_db")
    upload_to_gcs(bucket, "data", "data")
