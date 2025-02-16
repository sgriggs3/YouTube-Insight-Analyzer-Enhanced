from pathlib import Path
import re


def extract_google_keys():
    input_file = Path("found_gemini_keys.txt")
    output_file = Path("google_api_keys.txt")

    try:
        # Read content
        with open(input_file, "r") as f:
            content = f.readlines()

        # Extract AIza keys
        google_keys = {
            line.strip() for line in content if line.strip().startswith("AIza")
        }

        # Save unique Google API keys
        with open(output_file, "w") as f:
            f.write("=== Google API Keys ===\n\n")
            for key in sorted(google_keys):
                f.write(f"{key}\n")

        # Display results
        print(f"Found {len(google_keys)} Google API keys:")
        for key in sorted(google_keys):
            print(f"- {key}")

    except FileNotFoundError:
        print("Error: Input file not found")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    extract_google_keys()
