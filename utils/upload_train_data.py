import argparse
import base64
import boto3
import os
import botocore
from kubernetes import client, config
def upload_train_data(data_file, bucket_name, ns):
    # Load Kubernetes configuration
    config.load_kube_config()
    v1 = client.CoreV1Api()
    secret_name = "aws-connection-" + bucket_name
    print(f"Secret name: {secret_name}")
    try:
        secret = v1.read_namespaced_secret(secret_name, ns)
        #Decode bas64 encoded values
        secret_data = {key: base64.b64decode(value).decode('utf-8') for key, value in secret.data.items()}
    except client.ApiException as e:
        print(f"Error reading secret: {e}")
        return None
    print(f"Secret data: {secret_data}")
    aws_access_key_id = secret_data['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = secret_data['AWS_SECRET_ACCESS_KEY']
    endpoint_url = secret_data['AWS_S3_ENDPOINT']
    region_name = secret_data['AWS_DEFAULT_REGION']
    bucket_name = secret_data['AWS_S3_BUCKET']

#     Create session to S3 storage endpoint
    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    print(f"URL: {endpoint_url}")
    s3_resource = session.resource(
        's3',
        verify=False,
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)
    bucket = s3_resource.Bucket(bucket_name)

    print(f"Bucket name: {bucket_name}")
    upload_path  = os.path.join("train-data", os.path.basename(data_file))
    print(f"Uploading data to {upload_path}")
    bucket.upload_file(data_file, upload_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--bucket_name', type=str, required=True)
    parser.add_argument('--ns', type=str, required=True)
    args = parser.parse_args()
    upload_train_data(args.data_file, args.bucket_name, args.ns)
if __name__ == "__main__":
    main()

