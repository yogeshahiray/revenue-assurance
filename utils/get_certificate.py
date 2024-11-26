import argparse
import ssl

def get_certificate(site_name, port=443):
    cert = ssl.get_server_certificate((site_name,port))
    return cert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=str, required=True)
    args = parser.parse_args()
    print(f"Certificate for {args.site}")
    print(get_certificate(args.site))

if __name__ == "__main__":
    main()
