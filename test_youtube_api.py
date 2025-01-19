import youtube_api
import json

video_id = "dQw4w9WgXcQ"  # Replace with a valid video ID for testing

metadata = youtube_api.get_video_metadata(video_id)
print("Video Metadata:")
print(json.dumps(metadata, indent=2))

comments = youtube_api.get_video_comments(video_id)
print("\\nVideo Comments:")
print(json.dumps(comments, indent=2))

channel_data = youtube_api.get_channel_data("UC_x5XG1OVPErAZj1x6E9eWw")
print("\\nChannel Data:")
print(json.dumps(channel_data, indent=2))

transcript = youtube_api.get_video_transcript(video_id)
print("\\nVideo Transcript:")
print(json.dumps(transcript, indent=2))
