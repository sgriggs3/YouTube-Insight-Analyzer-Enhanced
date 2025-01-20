import requests
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import json
import time
import os
import logging
import googleapiclient.discovery
import random

logger = logging.getLogger(__name__)


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


def authenticate_youtube_api(api_key):
    return {"Authorization": f"Bearer {api_key}"}


def get_video_comments(video_id, scrape_type="latest", comment_limit=500):
    api_service_name = "youtube"
    api_version = "v3"
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=API_KEY
    )

    comments = []
    next_page_token = None
    total_comments = 0

    while total_comments < comment_limit:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, comment_limit - total_comments),
                pageToken=next_page_token,
                order="time" if scrape_type == "latest" else "relevance",
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                if scrape_type == "non_spam" and is_spam(comment):
                    continue
                comments.append(comment)
                total_comments += 1

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        except googleapiclient.errors.HttpError as e:
            logger.error(f"HTTP error while fetching comments: {e}")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Unexpected error while fetching comments: {e}")
            break

    if scrape_type == "random":
        comments = random.sample(comments, min(comment_limit, len(comments)))

    return comments


def is_spam(comment):
    # Implement spam detection logic here
    return False


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
        try:
            response = requests.get(url)
            if response.status_code == 200:
                metadata = response.json().get("items", [])
                return metadata
            elif response.status_code == 403:
                retries += 1
                time.sleep(2**retries)
            else:
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while fetching metadata: {e}")
            retries += 1
            time.sleep(2**retries)
    return []


def get_video_transcript(video_id, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript
        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
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
