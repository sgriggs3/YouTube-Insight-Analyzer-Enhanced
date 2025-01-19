import requests
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import json
import time
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import Dict, List, Optional, Any
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import cachetools


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


class YouTubeAPI:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or load_config().get("youtube_api_key")
        self.youtube = build("youtube", "v3", developerKey=self.api_key)
        self.cache = cachetools.TTLCache(maxsize=100, ttl=3600)  # 1 hour cache

        # Configure retry strategy
        self.session = requests.Session()
        retries = Retry(
            total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    @cachetools.cached(lambda self: self.cache)
    def get_video_metadata(self, video_id: str) -> Dict:
        """Fetch video metadata with caching and retry logic."""
        try:
            response = (
                self.youtube.videos()
                .list(part="snippet,statistics", id=video_id)
                .execute()
            )
            video = response["items"][0]
            return {
                "title": video["snippet"]["title"],
                "description": video["snippet"]["description"],
                "tags": video["snippet"].get("tags", []),
                "views": video["statistics"]["viewCount"],
                "likes": video["statistics"].get("likeCount", 0),
                "comments": video["statistics"].get("commentCount", 0),
            }
        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred: {e.content}")
            raise

    def get_video_comments(self, video_id: str, max_results: int = 100) -> List[Dict]:
        """Fetch video comments with pagination support."""
        try:
            comments = []
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(max_results, 100),
                textFormat="plainText",
            )

            while request and len(comments) < max_results:
                response = request.execute()

                for item in response["items"]:
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append(
                        {
                            "text": comment["textDisplay"],
                            "author": comment["authorDisplayName"],
                            "likes": comment["likeCount"],
                            "published_at": comment["publishedAt"],
                        }
                    )

                # Get next page of comments if available
                request = self.youtube.commentThreads().list_next(request, response)

                if len(comments) >= max_results:
                    break

            return comments[:max_results]

        except HttpError as e:
            print(f"An HTTP error {e.resp.status} occurred: {e.content}")
            raise

    def get_video_transcript(
        self, video_id: str, language_code: str = "en"
    ) -> Optional[str]:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=[language_code]
            )
            return " ".join([entry["text"] for entry in transcript])
        except Exception as e:
            logging.error(f"Error fetching transcript for video {video_id}: {e}")
            return None


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


def get_user_friendly_input():
    """
    Get user-friendly input options for URLs and feedback.
    """
    url = get_user_input("Enter the YouTube URL: ")
    feedback = get_user_input("Enter your feedback: ")
    return url, feedback
