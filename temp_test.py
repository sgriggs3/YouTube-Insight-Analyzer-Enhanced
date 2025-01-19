import test_system
import json

with open("config.json", "r") as f:
    config = json.load(f)

test_system.test_system_on_political_content(["M81aSGJq8jI"], config["youtube_api_key"])
