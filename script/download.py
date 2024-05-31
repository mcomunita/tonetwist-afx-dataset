import os
import wget
import yaml
import argparse
from urllib.parse import urlparse, urlunparse

def remove_query_from_url(url):
    parsed_url = urlparse(url)
    new_url = urlunparse(parsed_url._replace(query=''))
    return new_url

def get_filename_from_url(url):
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    return filename

def download_files(yaml_file, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    # Iterate over the dictionary
    for key, urls in data.items():
        print(f"Downloading files for key: {key}")
        for url in urls:
            clean_url = remove_query_from_url(url)
            filename = get_filename_from_url(clean_url)
            output_path = os.path.join(output_folder, filename)

            # Download the file using wget
            print(f"Downloading {url} to {output_path}")
            wget.download(url, out=output_path)
            print()  # For newline after wget progress

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from URLs specified in a YAML file.")
    parser.add_argument('yaml_file', type=str, help="Path to the YAML file containing URLs.")
    parser.add_argument('output_folder', type=str, help="Path to the output folder where files will be saved.")

    args = parser.parse_args()

    download_files(args.yaml_file, args.output_folder)