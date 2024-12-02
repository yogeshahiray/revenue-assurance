import argparse
import ssl
from kubernetes import client, config
import sys
def get_certificate(site_name, port=443):
    cert = ssl.get_server_certificate((site_name,port))
    return cert
def patch_dsci_cert(cert, dsci_name):
    ns = "istio-system"
    try:
        config.load_kube_config()
    except config.ConfigException:
        print("Failed to load Kubernetes config")
        sys.exit(1)
    api_instance = client.CustomObjectsApi()
    api_instance.patch_namespaced_custom_object("dscinitialization.opendatahub.io")

