import boto3
import os
import sys

BUCKET_NAME = "fruit-classifier-model-store"
MODEL_LOCAL_PATH = "fruit_ripeness_classifier/models/best_model.pth"
MODEL_S3_KEY = "models/best_model.pth"
REGION = "us-east-1"

def create_bucket_if_not_exists(s3, bucket_name, region):
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except Exception:
        print(f"Creating bucket '{bucket_name}'...")
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region}
        ) if region != "us-east-1" else s3.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created.")

def upload_model(local_path, bucket, key):
    if not os.path.exists(local_path):
        print(f"ERROR: Model file not found at '{local_path}'")
        print("Make sure you have best_model.pth in fruit_ripeness_classifier/models/")
        sys.exit(1)

    s3 = boto3.client("s3", region_name=REGION)
    create_bucket_if_not_exists(s3, bucket, REGION)

    print(f"Uploading {local_path} → s3://{bucket}/{key}")
    s3.upload_file(
        local_path,
        bucket,
        key,
        ExtraArgs={"ServerSideEncryption": "AES256"}
    )
    print("Upload complete.")
    print(f"Model is now at: s3://{bucket}/{key}")

if __name__ == "__main__":
    upload_model(MODEL_LOCAL_PATH, BUCKET_NAME, MODEL_S3_KEY)