import os
import json
import logging
import re
from typing import List, Dict
from pathlib import Path

# Configure logging with more verbose output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("api_collector.log")],
)
logger = logging.getLogger(__name__)


class APIKeyCollector:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.output_file = self.base_dir / "all_api_keys.txt"

        # Add all possible variations of Gemini key files
        self.log_files = [
            "gemini_api_keys.log",
            "gemini_api_key.txt",
            "gemini_keys.json",
            "gemini.key",
            ".gemini_keys",
            "gemini_api.key",
            "anthropic_keys.json",
            "lmstudio_keys.json",
            "openai_keys.json",
            "bedrock_keys.json",
            "vertex_keys.json",
        ]

    def read_api_keys(self, file_path: Path) -> List[str]:
        """Read API keys from a file with enhanced error handling."""
        try:
            if not file_path.exists():
                logger.debug(f"File not found: {file_path}")
                return []

            logger.info(f"Attempting to read: {file_path}")
            with open(file_path, "r") as file:
                if file_path.suffix == ".json":
                    try:
                        data = json.load(file)
                        if isinstance(data, dict):
                            keys = [v for v in data.values() if isinstance(v, str)]
                        elif isinstance(data, list):
                            keys = [item for item in data if isinstance(item, str)]
                        else:
                            keys = []
                        logger.info(f"Found {len(keys)} keys in JSON file: {file_path}")
                        return keys
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in file {file_path}: {e}")
                        # Try reading as plain text if JSON fails
                        file.seek(0)
                        return [
                            line.strip() for line in file.readlines() if line.strip()
                        ]
                else:
                    # For non-JSON files, read line by line
                    keys = [line.strip() for line in file.readlines() if line.strip()]
                    logger.info(f"Found {len(keys)} keys in text file: {file_path}")
                    return keys

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return []

    def collect_api_keys(self) -> Dict[str, List[str]]:
        """Collect API keys with improved logging."""
        collected_keys = {}
        logger.info(f"Searching for API keys in: {self.base_dir}")

        # Also check in common subdirectories
        search_dirs = [
            self.base_dir,
            self.base_dir / "config",
            self.base_dir / "keys",
            self.base_dir / ".keys",
            self.base_dir / "secrets",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            logger.info(f"Searching in directory: {search_dir}")
            for log_file in self.log_files:
                file_path = search_dir / log_file
                keys = self.read_api_keys(file_path)
                if keys:
                    collected_keys[log_file] = keys
                    logger.info(f"Found {len(keys)} keys in {log_file}")

        return collected_keys

    def save_api_keys(self, collected_keys: Dict[str, List[str]]) -> None:
        """Save collected API keys to output file."""
        try:
            with open(self.output_file, "w") as file:
                file.write("=== Collected API Keys ===\n\n")
                for source, keys in collected_keys.items():
                    file.write(f"\n= From {source} =\n")
                    for key in keys:
                        file.write(f"{key}\n")
            logger.info(f"API keys saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving API keys: {e}")


def extract_api_keys(log_file="gemini_api_keys.log"):
    try:
        with open(log_file, "r") as f:
            content = f.read()
            # Look for strings that match Gemini API key pattern
            # Typically they're alphanumeric strings
            keys = re.findall(r"[\w\-]{20,}", content)

            # Write found keys to output file
            with open("all_api_keys.txt", "w") as out:
                for key in keys:
                    out.write(f"{key}\n")

            print(f"Found {len(keys)} potential API keys")
            return keys
    except FileNotFoundError:
        print(f"Error: {log_file} not found")
        return []


def main():
    # Get directory from environment or use default
    directory = os.getenv(
        "API_KEYS_DIR", "/workspaces/Fix-my-prebui21YouTube-Insight-Analyzer-Enhanced"
    )

    try:
        logger.info("Starting API key collection")
        collector = APIKeyCollector(directory)

        # Create a test key if none exist (for testing)
        test_key_file = Path(directory) / "gemini_api_keys.log"
        if not test_key_file.exists():
            logger.info("Creating test key file")
            with open(test_key_file, "w") as f:
                f.write("test_api_key\n")

        collected_keys = collector.collect_api_keys()

        if collected_keys:
            collector.save_api_keys(collected_keys)
            logger.info(
                f"Successfully collected API keys from {len(collected_keys)} sources"
            )

            # Print found keys for verification (redacted)
            for source, keys in collected_keys.items():
                logger.info(f"Source: {source}, Keys found: {len(keys)}")
                for key in keys:
                    # Only show first/last 4 chars
                    redacted = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
                    logger.info(f"Found key: {redacted}")
        else:
            logger.warning("No API keys found in any of the source files")

    except Exception as e:
        logger.error(f"Failed to collect API keys: {e}")
        raise


if __name__ == "__main__":
    main()
    extract_api_keys()
