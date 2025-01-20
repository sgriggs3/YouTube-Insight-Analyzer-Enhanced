import asyncio
import logging
import os
from google.generativeai import GenerativeModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GeminiAPIServer:
    def __init__(self, api_key_log_path):
        self.api_key_log_path = api_key_log_path
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.model = GenerativeModel("gemini-pro")

    def _load_api_keys(self):
        if not os.path.exists(self.api_key_log_path):
            logging.error(f"API key log file not found: {self.api_key_log_path}")
            return []
        with open(self.api_key_log_path, "r") as f:
            keys = [line.strip() for line in f.readlines()]
        logging.info(f"Loaded {len(keys)} API keys from {self.api_key_log_path}")
        return keys

    def _get_current_api_key(self):
        if not self.api_keys:
            logging.error("No API keys available.")
            return None
        return self.api_keys[self.current_key_index]

    def _rotate_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logging.info(f"Rotated to API key index: {self.current_key_index}")

    async def call_gemini_api(self, prompt):
        api_key = self._get_current_api_key()
        if not api_key:
            return None
        try:
            self.model.api_key = api_key
            logging.info(f"Calling Gemini API with key index: {self.current_key_index}")
            response = await self.model.generate_content_async(prompt)
            logging.info(f"Gemini API call successful.")
            return response
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            logging.warning("Rate limit error detected, rotating API key.")
            self._rotate_api_key()
            return await self.call_gemini_api(prompt)  # Retry with the new key


async def main():
    api_key_log_path = "gemini_api_keys.log"  # Replace with your actual log file path
    server = GeminiAPIServer(api_key_log_path)
    prompt = "Write a short poem about the moon"
    response = await server.call_gemini_api(prompt)
    if response:
        print(f"Gemini API Response: {response.text}")
    else:
        print("Failed to get a response from Gemini API.")


if __name__ == "__main__":
    asyncio.run(main())
