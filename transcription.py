import whisper
from youtube_transcript_api import YouTubeTranscriptApi


import whisper
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp


def transcribe_youtube_video(video_id):
    try:
        try:
            model = whisper.load_model("base")
        except Exception as e:
            return f"Error loading whisper model: {e}"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "%(id)s.%(ext)s",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=True
                )
                audio_file = ydl.prepare_filename(info_dict)
        except Exception as e:
            return f"Error downloading audio: {e}"
        result = model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        return f"Error transcribing video: {e}"


def save_transcription_to_file(transcription, filename):
    try:
        with open(filename, "w") as f:
            f.write(transcription)
    except Exception as e:
        print(f"Error saving transcription to file: {e}")


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
