import argparse
import base64
import sys

from kubernetes import client, config

# Get the URL for Minio console
def get_minio_console_route(route_name, ns):
    # Load Kubernetes configuration
    try:
        config.load_kube_config()
    except config.ConfigException:
        print("Failed to load Kubernetes config")
        sys.exit(1)

    api_instance = client.CustomObjectsApi()
    route = api_instance.get_namespaced_custom_object("route.openshift.io", "v1", ns, "routes", route_name)
    host = route["spec"]["host"]
#     Return the full URL
    return f"https://{host}"

def get_minio_credentials(bucket_name, ns):
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
    aws_access_key_id = secret_data['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = secret_data['AWS_SECRET_ACCESS_KEY']
    endpoint_url = secret_data['AWS_S3_ENDPOINT']
    region_name = secret_data['AWS_DEFAULT_REGION']
    bucket_name = secret_data['AWS_S3_BUCKET']

    console_route = get_minio_console_route("minio-console", ns)
    print(f"Minio Console URL: {console_route}")
    print(f"Minio username: {aws_access_key_id}")
    print(f"Minio password: {aws_secret_access_key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name', type=str, required=True)
    parser.add_argument('--ns', type=str, required=True)
    args = parser.parse_args()
    get_minio_credentials(args.bucket_name, args.ns)
if __name__ == "__main__":
    main()

