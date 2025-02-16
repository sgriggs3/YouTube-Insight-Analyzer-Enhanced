from pathlib import Path


def extract_unique_keys():
    # Input and output file paths
    input_file = Path("gemini_api_keys.log")
    output_file = Path("unique_gemini_keys.txt")

    try:
        # Read and get unique keys
        with open(input_file, "r") as f:
            # Use set to automatically remove duplicates
            unique_keys = {line.strip() for line in f if line.strip()}

        # Save unique keys
        with open(output_file, "w") as f:
            for key in unique_keys:
                f.write(f"{key}\n")

        # Display results
        print(f"Found {len(unique_keys)} unique keys:")
        for key in unique_keys:
            print(f"- {key}")

    except FileNotFoundError:
        print("Error: gemini_api_keys.log not found")
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    extract_unique_keys()
