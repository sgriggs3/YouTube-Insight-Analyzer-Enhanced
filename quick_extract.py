from pathlib import Path
import sys


def quick_extract():
    log_file = Path("gemini_api_keys.log")
    output_file = Path("extracted_keys.txt")

    try:
        # Read all content from log file
        with open(log_file, "r") as f:
            content = f.readlines()

        # Remove duplicates while preserving order
        unique_content = dict.fromkeys(content)

        # Write to output file
        with open(output_file, "w") as f:
            f.writelines(unique_content)

        # Display results
        print(f"Extracted {len(unique_content)} unique entries")
        print("\nContents:")
        print("=========")
        sys.stdout.writelines(unique_content)

    except FileNotFoundError:
        print("Error: gemini_api_keys.log not found!")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    quick_extract()
