import asyncio
import logging
import json
import aiohttp
from aiohttp import ClientError
import subprocess
import os
import time

# 1. Initialization
log_file = "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_file,
    filemode="w",
)
logger = logging.getLogger(__name__)

api_keys = ["key1", "key2", "key3"]  # Replace with your actual API keys
current_api_key_index = 0
backend_api_url = "http://backend.example.com/api"  # Replace with your backend API URL


# 2. Web UI Deployment
def start_web_ui():
    """Starts the React app's development server."""
    try:
        logger.info("Starting React app development server...")
        # Start the React app using npm start in the frontend directory
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=os.path.join(os.getcwd(), "frontend"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Wait for the server to start (look for "started server on" in output)
        while True:
            line = process.stdout.readline()
            if "started server on" in line.lower():
                logger.info("React app development server started.")
                break
            if not line:
                # Check if the process has terminated
                if process.poll() is not None:
                    logger.error(
                        f"React app development server failed to start. Error: {process.stderr.read()}"
                    )
                    return None
            time.sleep(0.1)
        return "http://localhost:3000"  # Default React dev server port
    except Exception as e:
        logger.error(f"Error starting web UI: {e}")
        return None


# 3. Backend Synchronization Check
class RateLimitError(Exception):
    pass


async def check_backend_sync():
    try:
        data = await fetch_backend_data()
        if data_is_synchronized(data):
            logger.info("Backend synchronization successful")
            return True
        else:
            logger.error("Backend synchronization failed: Data mismatch")
            return False
    except Exception as e:
        logger.error(f"Error during backend synchronization check: {e}")
        return False


async def fetch_backend_data():
    while True:
        try:
            api_key = api_keys[current_api_key_index]
            logger.info(f"API Request: GET {backend_api_url}/status with key {api_key}")
            response = await make_api_request(f"{backend_api_url}/status", api_key)
            logger.info(f"API Response: {response}")
            return response
        except RateLimitError:
            logger.warning("Rate limit error encountered. Rotating API key.")
            rotate_api_key()
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise e


async def make_api_request(url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                if response.status == 429:
                    raise RateLimitError("Rate limit exceeded")
                return await response.json()
        except aiohttp.ClientError as e:
            raise Exception(f"Client error during API request: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"JSON decode error: {e}")
        except Exception as e:
            raise e


def rotate_api_key():
    global current_api_key_index
    current_api_key_index = (current_api_key_index + 1) % len(api_keys)


def data_is_synchronized(data):
    # Implement logic to check if data is synchronized
    # Example: return data.get("status") == "ok"
    return True  # Placeholder


# 4. Web UI Testing
def start_web_ui_test(web_ui_url):
    logger.info(f"Web UI deployed at: {web_ui_url}")
    logger.info("Please manually test the web UI, focusing on:")
    logger.info("- Data display")
    logger.info("- User interactions")
    logger.info("Report any errors or issues encountered.")


# Main execution
async def main():
    web_ui_url = start_web_ui()
    if web_ui_url:
        await asyncio.sleep(1)  # Give the server time to start
        sync_successful = await check_backend_sync()
        if sync_successful:
            start_web_ui_test(web_ui_url)
        else:
            logger.error("Backend synchronization failed. Web UI testing not started.")
    else:
        logger.error("Web UI failed to start. Backend sync and testing not started.")


if __name__ == "__main__":
    asyncio.run(main())
