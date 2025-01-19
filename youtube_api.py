import requests
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import json
import time
import os


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def authenticate_youtube_api(api_key):
    return {"Authorization": f"Bearer {api_key}"}


def get_video_comments(video_id, options=None):
    config = load_config()
    api_key = config.get("youtube_api_key")
    comments = []
    next_page_token = None

    # Default options
    default_options = {"limit": 1000, "sort_by": "relevance", "include_replies": True}
    options = {**default_options, **(options or {})}

    while True:
        try:
            url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={api_key}&maxResults=100"
            if next_page_token:
                url += f"&pageToken={next_page_token}"

            # Add sorting options
            if options["sort_by"] == "time":
                url += "&order=time"

            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                comments.extend(data.get("items", []))

                # Save progress
                save_scraping_progress(video_id, len(comments), options["limit"])

                next_page_token = data.get("nextPageToken")
                if not next_page_token or len(comments) >= options["limit"]:
                    break

            elif response.status_code == 403:
                handle_rate_limit()
            else:
                handle_error(response)
                break

        except Exception as e:
            log_error(e)
            break

    return comments


def save_scraping_progress(video_id, current_count, total):
    """Save scraping progress to finished-tracks.txt"""
    with open("finished-tracks.txt", "a") as f:
        f.write(f"Video {video_id}: {current_count}/{total} comments scraped\n")


def get_video_metadata(video_id, max_retries=5):
    config = load_config()
    api_key = config.get("youtube_api_key")
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&id={video_id}&key={api_key}"
    retries = 0
    while retries < max_retries:
        response = requests.get(url)
        if response.status_code == 200:
            metadata = response.json().get("items", [])
            return metadata
        elif response.status_code == 403:
            retries += 1
            time.sleep(2**retries)
        else:
            return []
    return []


def get_video_transcript(video_id, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript
        except Exception as e:
            retries += 1
            time.sleep(2**retries)
    return []


def save_data_to_csv(data, filename):
    config = load_config()
    data_storage_path = config.get("data_storage_path", "data")
    os.makedirs(data_storage_path, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_storage_path, filename), index=False)


def save_data_to_json(data, filename):
    config = load_config()
    data_storage_path = config.get("data_storage_path", "data")
    os.makedirs(data_storage_path, exist_ok=True)
    with open(os.path.join(data_storage_path, filename), "w") as f:
        json.dump(data, f)


def load_data_from_json(filename):
    config = load_config()
    data_storage_path = config.get("data_storage_path", "data")
    try:
        with open(os.path.join(data_storage_path, filename), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL.
    """
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        return None


def handle_user_feedback(feedback, sentiment_data):
    """
    Incorporate user feedback into the sentiment data.
    """
    for item in feedback:
        comment_id = item.get("comment_id")
        corrected_sentiment = item.get("corrected_sentiment")
        sentiment_data.loc[
            sentiment_data["comment_id"] == comment_id, "corrected_sentiment"
        ] = corrected_sentiment
    return sentiment_data


def get_user_input(prompt):
    """
    Get user input from the command line.
    """
    return input(prompt)


def save_data_to_csv(data, filename):
    """
    Save data to a CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)


def get_user_friendly_input():
    """
    Get user-friendly input options for URLs and feedback.
    """
    url = get_user_input("Enter the YouTube URL: ")
    feedback = get_user_input("Enter your feedback: ")
    return url, feedback


def save_data_to_csv(data, filename):
    """
    Save data to a CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
