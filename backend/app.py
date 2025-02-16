from flask import Flask, jsonify, request
from flask_cors import CORS
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


@app.route("/api/comments")
def get_comments_route():
    video_id = request.args.get("videoId")
    max_results = request.args.get("maxResults", default=500, type=int)
    comments = get_video_comments(video_id, comment_limit=max_results)
    return jsonify(
        [
            {
                "text": comment,
                "author": "Unknown",
                "timestamp": "Unknown",
                "sentiment": "Unknown",
            }
            for comment in comments
        ]
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
