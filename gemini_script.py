import argparse
import json
import logging
from multiprocessing import Pool
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Default API configuration
BASE_URL = "https://api.gemini.com/v1"
ENDPOINTS = ["balances", "orders", "trades"]

# Fetch data from a specific endpoint using an API key
def fetch_data(args):
    api_key, endpoint = args
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        data = response.json()
        logging.info(f"Success: Fetched data from {endpoint} with API key {api_key}")
        return {api_key: {endpoint: data}}
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from {endpoint} with API key {api_key}: {e}")
        return {api_key: {endpoint: None}}

# Main function to distribute tasks
def main(api_keys, output_file):
    # Prepare arguments for multiprocessing
    tasks = [(api_key, endpoint) for api_key in api_keys for endpoint in ENDPOINTS]

    # Use a process pool to fetch data concurrently
    with Pool(len(api_keys)) as pool:
        results = pool.map(fetch_data, tasks)

    # Combine results
    combined_results = {}
    for result in results:
        for api_key, data in result.items():
            if api_key not in combined_results:
                combined_results[api_key] = {}
            combined_results[api_key].update(data)

    # Save results to a file
    with open(output_file, "w") as f:
        json.dump(combined_results, f, indent=4)

    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch data from Gemini API using multiprocessing")
    parser.add_argument("--api-keys", nargs="+", required=True, help="List of Gemini API keys")
    parser.add_argument("--output-file", required=True, help="File to save the results (e.g., results.json)")
    args = parser.parse_args()

    api_keys = os.getenv("API_KEYS").split(",")
    main(api_keys, args.output_file)
