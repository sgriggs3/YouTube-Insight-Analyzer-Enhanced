from flask import Flask, jsonify, request
from flask_cors import CORS
import re
from data_visualization import (
    visualize_sentiment_trends,
    create_word_cloud,
    create_sentiment_distribution_chart,
    visualize_user_engagement,
)
from youtube_api import get_video_metadata, get_video_comments

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return "Backend is running!"


@app.route("/api/sentiment/trends", methods=["POST"])
def get_sentiment_trends():
    data = request.json
    result = visualize_sentiment_trends(data["sentiment_data"], "sentiment_trends.html")
    return jsonify({"status": "success", "file": "sentiment_trends.html"})


@app.route("/api/wordcloud", methods=["POST"])
def get_word_cloud():
    data = request.json
    create_word_cloud(data["text_data"], "wordcloud.png")
    return jsonify({"status": "success", "file": "wordcloud.png"})


@app.route("/api/sentiment/distribution", methods=["POST"])
def get_sentiment_distribution():
    data = request.json
    create_sentiment_distribution_chart(data["sentiment_data"], "distribution.html")
    return jsonify({"status": "success", "file": "distribution.html"})


@app.route("/api/engagement", methods=["POST"])
def get_user_engagement():
    data = request.json
    visualize_user_engagement(data["engagement_data"], "engagement.html")
    return jsonify({"status": "success", "file": "engagement.html"})


@app.route("/api/video-metadata/<video_id>")
def get_video_metadata_route(video_id):
    metadata = get_video_metadata(video_id)
    if metadata:
        if metadata and len(metadata) > 0:
            video_data = metadata[0]
            return jsonify(
                {
                    "title": video_data["snippet"]["title"],
                    "description": video_data["snippet"]["description"],
                    "views": int(video_data["statistics"]["viewCount"]),
                    "likes": int(video_data["statistics"]["likeCount"]),
                    "publishedAt": video_data["snippet"]["publishedAt"],
                }
            )
        else:
            return jsonify({"error": "Video metadata not found"}), 404
    else:
        return jsonify({"error": "Failed to fetch video metadata"}), 500


def get_comments_route():
    url_or_video_id = request.args.get("urlOrVideoId")
    max_results = request.args.get("maxResults", default=500, type=int)
    video_id = None
    if url_or_video_id:
        # Regex to extract videoId from YouTube URL
        match = re.search(r"(?:v=|\/embed\/|\/watch\?v=|\/shorts\/|youtu\.be\/)([\w-]{11})", url_or_video_id)
        if match:
            video_id = match.group(1)
        else:
            video_id = url_or_video_id  # Assume it's already a videoId

    if not video_id:
        return jsonify({"error": "No videoId or valid YouTube URL provided"}), 400

    comments = get_video_comments(video_id, comment_limit=max_results)
    return jsonify(comments)

@app.route("/api/comments/csv")
def get_comments_csv_route():
    video_id = request.args.get("urlOrVideoId")
    max_results = request.args.get("maxResults", default=500, type=int)
    video_id_extracted = None
    if video_id:
        match = re.search(r"(?:v=|\/embed\/|\/watch\?v=|\/shorts\/|youtu\.be\/)([\w-]{11})", video_id)
        if match:
            video_id_extracted = match.group(1)
        else:
            video_id_extracted = video_id

    if not video_id_extracted:
        return jsonify({"error": "No videoId or valid YouTube URL provided"}), 400

    comments = get_video_comments(video_id_extracted, comment_limit=max_results)
    
    csv_filename = f"comments_{video_id_extracted}.csv"
    csv_content = "Comment\\n"  # Header
    for comment in comments:
        csv_content += f"{comment['comment'].replace('\"', '').replace(',', ';')}\\n" # Basic CSV formatting, replace quotes and commas

    return jsonify({"status": "success", "file": csv_filename, "csv_content": csv_content})


@app.route("/api/sentiment-analysis")
def get_sentiment_analysis_route():
    video_id = request.args.get("urlOrVideoId")
    max_results = request.args.get("maxResults", default=500, type=int)
    video_id_extracted = None
    if video_id:
        match = re.search(r"(?:v=|\/embed\/|\/watch\?v=|\/shorts\/|youtu\.be\/)([\w-]{11})", video_id)
        if match:
            video_id_extracted = match.group(1)
        else:
            video_id_extracted = video_id

    if not video_id_extracted:
        return jsonify({"error": "No videoId or valid YouTube URL provided"}), 400

    comments_data = get_video_comments(video_id_extracted, comment_limit=max_results)
    if not comments_data or not isinstance(comments_data, list):
        return jsonify({"error": "Failed to fetch comments for sentiment analysis"}), 500

    sentiment_results = []
    for comment_item in comments_data:
        comment_text = comment_item.get('comment', '')
        if comment_text:
            # Basic sentiment analysis - replace with actual sentiment analysis logic later
            sentiment = "positive" if len(comment_text) % 2 == 0 else "negative" # Placeholder
            sentiment_results.append({"comment": comment_text, "sentiment": sentiment})

    return jsonify({"status": "success", "sentiment_results": sentiment_results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
