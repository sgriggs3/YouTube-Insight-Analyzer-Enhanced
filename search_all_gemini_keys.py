from pathlib import Path
import re


def search_all_files_for_keys():
    # Common patterns for Gemini API keys
    KEY_PATTERNS = [
        r"AIza[0-9A-Za-z\-_]{35}",  # Google API key pattern
        r"[\w\-]{39}",  # Gemini API key pattern
        r"gemini[0-9A-Za-z\-_]{32}",  # Alternative Gemini pattern
        r'api_key["\s]*=["\s]*([0-9A-Za-z\-_]{35,40})',  # Key assignment pattern
    ]

    base_dir = Path("/workspaces/Fix-my-prebui21YouTube-Insight-Analyzer-Enhanced")
    found_keys = set()  # Using set for automatic deduplication

    # Files to skip
    SKIP_DIRS = {".git", "node_modules", "__pycache__", "dist", "build"}

    def should_process(path):
        return not any(skip_dir in path.parts for skip_dir in SKIP_DIRS)

    print("Searching for Gemini API keys...")

    # Recursively search all files
    for file_path in base_dir.rglob("*"):
        if file_path.is_file() and should_process(file_path):
            try:
                content = file_path.read_text()
                for pattern in KEY_PATTERNS:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        key = match.group()
                        if len(key) >= 35:  # Minimum length for valid key
                            found_keys.add(key)
                            print(f"Found potential key in: {file_path}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Save unique keys
    output_file = base_dir / "found_gemini_keys.txt"
    with open(output_file, "w") as f:
        f.write("=== Found Gemini API Keys ===\n\n")
        for key in sorted(found_keys):
            f.write(f"{key}\n")

    print(f"\nFound {len(found_keys)} unique keys")
    print(f"Saved to: {output_file}")

    # Display found keys (redacted)
    print("\nFound keys (redacted):")
    for key in sorted(found_keys):
        redacted = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
        print(f"- {redacted}")


if __name__ == "__main__":
    search_all_files_for_keys()
