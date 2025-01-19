import requests
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import json
import time

def authenticate_youtube_api(api_key):
    return {"Authorization": f"Bearer {api_key}"}

def get_video_comments(video_id, api_key, max_retries=5):
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={api_key}"
    retries = 0
    while retries < max_retries:
        response = requests.get(url)
        if response.status_code == 200:
            comments = response.json().get("items", [])
            return comments
        elif response.status_code == 403:
            retries += 1
            time.sleep(2 ** retries)
        else:
            return []
    return []

def get_video_metadata(video_id, api_key, max_retries=5):
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,statistics&id={video_id}&key={api_key}"
    retries = 0
    while retries < max_retries:
        response = requests.get(url)
        if response.status_code == 200:
            metadata = response.json().get("items", [])
            return metadata
        elif response.status_code == 403:
            retries += 1
            time.sleep(2 ** retries)
        else:
            return []
    return []

def save_data_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def save_data_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

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
        comment_id = item.get('comment_id')
        corrected_sentiment = item.get('corrected_sentiment')
        sentiment_data.loc[sentiment_data['comment_id'] == comment_id, 'corrected_sentiment'] = corrected_sentiment
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
