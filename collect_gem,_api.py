import os

# Define the directory to search for API key log files
directory = "/workspaces/Fix-my-prebui21YouTube-Insight-Analyzer-Enhanced"

# Define the output file
output_file = os.path.join(directory, "all_api_keys.txt")

# List of log files to search for API keys
log_files = [
    "gemi    FLASK_APP=run.py
    FLASK_ENV=development
    FLASK_DEBUG=true
    YOUTUBE_API_KEY=your_api_key_here ni_api_keys.log",
    "anthropic_keys.json",
    "lmstudio_keys.json",
    "openai_keys.json",
    "bedrock_keys.json",
    "vertex_keys.json"
]

# Function to read API keys from a file
def read_api_keys(file_path):
    try:
        with open(file_path, "r") as file:
            return file.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# Collect all API keys
all_api_keys = []
for log_file in log_files:
    file_path = os.path.join(directory, log_file)
    all_api_keys.extend(read_api_keys(file_path))

# Save all API keys to the output file
with open(output_file, "w") as file:
    for key in all_api_keys:
        file.write(key)

print(f"All API keys have been saved to {output_file}")

# Flask environment variables
FLASK_APP = "run.py"
FLASK_ENV = "development"
FLASK_DEBUG = True
YOUTUBE_API_KEY = "your_api_key_here"
