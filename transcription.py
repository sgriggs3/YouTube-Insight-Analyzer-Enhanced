import whisper
from youtube_transcript_api import YouTubeTranscriptApi

def transcribe_youtube_video(video_id):
    model = whisper.load_model("base")
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([item['text'] for item in transcript])
    result = model.transcribe(text)
    return result['text']

def save_transcription_to_file(transcription, filename):
    with open(filename, 'w') as f:
        f.write(transcription)

def transcribe_with_user_input(video_id, user_feedback=None):
    transcription = transcribe_youtube_video(video_id)
    if user_feedback:
        transcription += "\n\nUser Feedback:\n" + user_feedback
    return transcription

def save_transcription_with_feedback(transcription, filename, user_feedback=None):
    with open(filename, 'w') as f:
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
