import whisper
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_api import YouTubeAPI, save_data_to_json
import logging


def transcribe_youtube_video(video_id):
    model = whisper.load_model("base")
    try:
        transcription = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([item["text"] for item in transcription])
        result = model.transcribe(text)
        return result["text"]
    except Exception as e:
        logging.error(f"Transcription failed for video {video_id}: {e}")
        return ""


def save_transcription_to_file(transcription, filename):
    with open(filename, "w") as f:
        f.write(transcription)


def transcribe_with_user_input(video_id, user_feedback=None):
    transcription = transcribe_youtube_video(video_id)
    if user_feedback:
        transcription += "\n\nUser Feedback:\n" + user_feedback
    return transcription


def save_transcription_with_feedback(transcription, filename, user_feedback=None):
    with open(filename, "w") as f:
        f.write(transcription)
        if user_feedback:
            f.write("\n\nUser Feedback:\n" + user_feedback)


def get_user_input(prompt):
    return input(prompt)


def transcribe_with_user_input_and_feedback(video_id):
    user_feedback = get_user_input("Enter your feedback: ")
    transcription = transcribe_with_user_input(video_id, user_feedback)
    return transcription


def save_transcription_with_user_input_and_feedback(transcription, filename):
    user_feedback = get_user_input("Enter your feedback: ")
    save_transcription_with_feedback(transcription, filename, user_feedback)


def fetch_and_store_transcript(video_id: str):
    """
    Fetch transcript for a video and store it as JSON.
    """
    youtube_api = YouTubeAPI()
    transcript = youtube_api.get_video_transcript(video_id)
    if transcript:
        save_data_to_json(transcript, f"{video_id}_transcript.json")
    else:
        logging.warning(f"No transcript available for video {video_id}")
