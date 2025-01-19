import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import youtube_api
import sentiment_analysis
import data_visualization
from data_visualization import start_websocket_server, real_time_visualization
import asyncio
import websockets
from datetime import datetime
from functools import lru_cache
import redis
import logging

app = Flask(__name__)

# Setup Redis cache
redis_client = redis.Redis(host="localhost", port=6379, db=0)
CACHE_TIMEOUT = 3600  # 1 hour


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


config = load_config()
web_ui_port = config.get("web_ui_port", 5000)

# Load data
data = pd.read_csv("sentiment_data.csv")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/filter", methods=["POST"])
def filter_data():
    criteria = request.json
    filtered_data = data
    for key, value in criteria.items():
        filtered_data = filtered_data[filtered_data[key] == value]
    fig = px.line(
        filtered_data, x="date", y="sentiment", title="Filtered Sentiment Analysis"
    )
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify(graph_json)


@app.route("/real-time")
def real_time():
    async def connect_websocket():
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            await real_time_visualization(websocket, "/")

    asyncio.run(connect_websocket())
    return "Real-time visualization started"


@app.route("/input-url", methods=["POST"])
def input_url():
    url = request.json.get("url")
    try:
        video_id = youtube_api.extract_video_id(url)
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400
        comments = youtube_api.get_video_comments(video_id)
        sentiment_results = sentiment_analysis.perform_sentiment_analysis(
            [comment["text"] for comment in comments]
        )
        youtube_api.save_data_to_csv(
            sentiment_results, f"{video_id}_sentiment_analysis.csv"
        )
        return jsonify(
            {
                "message": "URL processed successfully",
                "video_id": video_id,
                "sentiment_results": sentiment_results,
            }
        )
    except Exception as e:
        return (
            jsonify({"message": f"Error processing URL: {str(e)}", "video_id": None}),
            500,
        )


@app.route("/user-feedback", methods=["POST"])
def user_feedback():
    feedback = request.json.get("feedback")
    sentiment_data = pd.read_csv("sentiment_data.csv")
    updated_sentiment_data = sentiment_analysis.incorporate_user_feedback(
        feedback, sentiment_data
    )
    updated_sentiment_data.to_csv("sentiment_data.csv", index=False)
    return jsonify({"message": "Feedback received successfully"})


@app.route("/save-csv", methods=["POST"])
def save_csv():
    data_to_save = request.json.get("data")
    filename = request.json.get("filename", "output.csv")
    df = pd.DataFrame(data_to_save)
    df.to_csv(filename, index=False)
    return jsonify({"message": "Data saved to CSV successfully", "filename": filename})


@app.route("/configuration")
def configuration():
    return render_template("configuration.html")


@app.route("/save-configuration", methods=["POST"])
def save_configuration():
    config = request.json
    # In a real application, you would save the configuration to a file or database
    # For this example, we'll just print the configuration
    print("Configuration:", config)
    return jsonify({"message": "Configuration saved successfully"})


@app.route("/scraping")
def scraping():
    return render_template("scraping.html")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/examples")
def examples():
    return render_template("examples.html")


@app.route("/video-metadata", methods=["GET"])
def video_metadata():
    video_id = request.args.get("video_id")
    return render_template("video_metadata.html", video_id=video_id)


@app.route("/sentiment-results", methods=["GET"])
def sentiment_results():
    video_id = request.args.get("video_id")
    return render_template("sentiment_results.html", video_id=video_id)


@app.route("/visualization/heatmap", methods=["GET"])
def visualization_heatmap():
    data = pd.read_csv("sentiment_data.csv")
    criteria = request.args.to_dict()
    if criteria:
        data = data_visualization.filter_data(data, criteria)
    filename = "heatmap.html"
    data_visualization.create_heatmap(data, filename)
    return render_template("visualization.html", visualization_file=filename)


@app.route("/visualization/wordcloud", methods=["GET"])
def visualization_wordcloud():
    data = pd.read_csv("sentiment_data.csv")
    criteria = request.args.to_dict()
    if criteria:
        data = data_visualization.filter_data(data, criteria)
    filename = "wordcloud.html"
    data_visualization.create_word_cloud(data, filename)
    return render_template("visualization.html", visualization_file=filename)


@app.route("/visualization/sentiment-distribution", methods=["GET"])
def visualization_sentiment_distribution():
    data = pd.read_csv("sentiment_data.csv")
    criteria = request.args.to_dict()
    if criteria:
        data = data_visualization.filter_data(data, criteria)
    filename = "sentiment_distribution.html"
    data_visualization.create_sentiment_distribution_chart(data, filename)
    return render_template("visualization.html", visualization_file=filename)


@app.route("/input-url-form", methods=["GET", "POST"])
def input_url_form():
    if request.method == "POST":
        url = request.form.get("url")
        video_id = youtube_api.extract_video_id(url)
        comments = youtube_api.get_video_comments(video_id)
        sentiment_results = sentiment_analysis.perform_sentiment_analysis(
            [comment["text"] for comment in comments]
        )
        youtube_api.save_data_to_csv(
            sentiment_results, f"{video_id}_sentiment_analysis.csv"
        )
        return jsonify({"message": "URL processed successfully", "video_id": video_id})
    return render_template("input_url_form.html")


@app.route("/start-sentiment-analysis", methods=["POST"])
def start_sentiment_analysis():
    urls = request.json.get("urls")
    options = request.json.get("options")
    results = []
    for url in urls:
        try:
            video_id = youtube_api.extract_video_id(url)
            comments = youtube_api.get_video_comments(video_id)
            sentiment_results = sentiment_analysis.perform_sentiment_analysis(
                [comment["text"] for comment in comments]
            )
            results.append(sentiment_results)
        except Exception as e:
            results.append({"error": f"Error processing URL: {url}, {str(e)}"})
    return jsonify(
        {"message": "Sentiment analysis started successfully", "results": results}
    )


@app.route("/dynamic-suggestions", methods=["GET"])
def dynamic_suggestions():
    sentiment_data = pd.read_csv("sentiment_data.csv")
    suggestions = sentiment_analysis.generate_dynamic_suggestions(sentiment_data)
    return jsonify({"suggestions": suggestions})


@app.route("/user-feedback-form", methods=["GET", "POST"])
def user_feedback_form():
    if request.method == "POST":
        feedback = request.form.get("feedback")
        sentiment_data = pd.read_csv("sentiment_data.csv")
        updated_sentiment_data = sentiment_analysis.incorporate_user_feedback(
            feedback, sentiment_data
        )
        updated_sentiment_data.to_csv("sentiment_data.csv", index=False)
        return jsonify({"message": "Feedback received successfully"})
    return render_template("user_feedback_form.html")


@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/api/video-metadata/<video_id>", methods=["GET"])
def api_video_metadata(video_id):
    metadata = youtube_api.get_video_metadata(video_id)
    return jsonify(metadata)


@app.route("/api/sentiment-analysis/<video_id>", methods=["GET"])
def api_sentiment_analysis(video_id):
    comments = youtube_api.get_video_comments(video_id)
    transcript = youtube_api.get_video_transcript(video_id)

    text_inputs = []
    if comments:
        text_inputs.extend(
            [
                {
                    "text": comment["text"],
                    "type": "comment",
                }
                for comment in comments
            ]
        )
    if transcript:
        text_inputs.extend(
            [{"text": item["text"], "type": "transcript"} for item in transcript]
        )

    sentiment_results = sentiment_analysis.perform_sentiment_analysis(text_inputs)
    return jsonify(sentiment_results)


@app.route("/api/visualizations/<video_id>", methods=["GET"])
def api_visualizations(video_id):
    comments = youtube_api.get_video_comments(video_id)
    transcript = youtube_api.get_video_transcript(video_id)

    text_inputs = []
    if comments:
        text_inputs.extend(
            [
                {
                    "text": comment["text"],
                    "type": "comment",
                }
                for comment in comments
            ]
        )
    if transcript:
        text_inputs.extend(
            [{"text": item["text"], "type": "transcript"} for item in transcript]
        )

    sentiment_results = sentiment_analysis.perform_sentiment_analysis(text_inputs)

    output_file = f"{video_id}_visualizations"
    data_visualization.visualize_sentiment_by_type(sentiment_results, output_file)

    return jsonify(
        {"message": "Visualizations generated successfully", "output_file": output_file}
    )


@lru_cache(maxsize=100)
def get_cached_video_metadata(video_id: str) -> Dict[str, Any]:
    """Cache video metadata in memory and Redis."""
    # Try Redis first
    cache_key = f"metadata:{video_id}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Fetch from API if not cached
    metadata = youtube_api.get_video_metadata(video_id)
    redis_client.setex(cache_key, CACHE_TIMEOUT, json.dumps(metadata))
    return metadata


class WebSocketManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket):
        self.active_connections.add(websocket)
        try:
            video_id = youtube_api.extract_video_id(url)
            if not video_id:
                return jsonify({"error": "Invalid YouTube URL"}), 400

            # Parallel fetching of metadata and comments
            metadata_task = asyncio.create_task(get_cached_video_metadata(video_id))
            comments_task = asyncio.create_task(
                youtube_api.get_video_comments(video_id)
            )

            metadata, comments = await asyncio.gather(metadata_task, comments_task)

            # Process sentiment analysis in background
            asyncio.create_task(process_sentiment_analysis(video_id, comments))

            return jsonify(
                {
                    "success": True,
                    "metadata": metadata,
                    "comments": comments[:10],
                    "commentCount": len(comments),
                    "job_id": video_id,  # For polling sentiment results
                }
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """API endpoint to analyze a video URL"""
    try:
        url = request.json.get("url")
        if not url:
            return jsonify({"error": "No URL provided"}), 400

        video_id = youtube_api.extract_video_id(url)
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        api = youtube_api.YouTubeAPI()

        # Get video metadata
        try:
            metadata = api.get_video_metadata(video_id)
        except Exception as e:
            return jsonify({"error": f"Failed to get video metadata: {str(e)}"}), 500

        # Get video comments
        try:
            comments = api.get_video_comments(video_id)
        except Exception as e:
            return jsonify({"error": f"Failed to get video comments: {str(e)}"}), 500

        # Save data
        youtube_api.save_data_to_json(metadata, f"{video_id}_metadata.json")
        youtube_api.save_data_to_csv(comments, f"{video_id}_comments.csv")

        return jsonify(
            {
                "success": True,
                "metadata": metadata,
                "comments": comments[:10],  # First 10 comments for preview
                "commentCount": len(comments),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sentiment", methods=["POST"])
def analyze_sentiment():
    """API endpoint for sentiment analysis"""
    try:
        video_id = request.json.get("video_id")
        if not video_id:
            return jsonify({"error": "No video ID provided"}), 400

        comments_file = f"data/{video_id}_comments.csv"
        if not os.path.exists(comments_file):
            return jsonify({"error": "Comments data not found"}), 404

        # Load and analyze comments
        comments_df = pd.read_csv(comments_file)

        # TODO: Add actual sentiment analysis
        sentiment_results = {"positive": 0, "negative": 0, "neutral": 0}

        return jsonify({"success": True, "sentiment": sentiment_results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


import threading


async def main():
    start_server = websockets.serve(real_time_visualization, "localhost", 8765)
    flask_thread = threading.Thread(
        target=app.run,
        kwargs={"debug": True, "port": web_ui_port, "use_reloader": False},
    )
    flask_thread.start()
    await start_server


if __name__ == "__main__":
    asyncio.run(main())

from flask import Flask, request, jsonify
from youtube_api import YouTubeAPI
from sentiment_analysis import SentimentAnalyzer
from data_visualization import visualize_sentiment_trends

app = Flask(__name__)
youtube_api = YouTubeAPI()
sentiment_analyzer = SentimentAnalyzer()


@app.route("/api/video/metadata", methods=["GET"])
def get_video_metadata():
    video_id = request.args.get("video_id")
    if not video_id:
        return jsonify({"error": "video_id parameter is required"}), 400
    try:
        metadata = youtube_api.get_video_metadata(video_id)
        return jsonify(metadata)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/comments", methods=["GET"])
def get_video_comments():
    video_id = request.args.get("video_id")
    max_results = request.args.get("max_results", default=100, type=int)
    if not video_id:
        return jsonify({"error": "video_id parameter is required"}), 400
    try:
        comments = youtube_api.get_video_comments(video_id, max_results)
        analyzed_comments = sentiment_analyzer.analyze_comments(comments)
        return jsonify(analyzed_comments)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/video/transcript", methods=["GET"])
def get_video_transcript():
    video_id = request.args.get("video_id")
    if not video_id:
        return jsonify({"error": "video_id parameter is required"}), 400
    try:
        transcript = youtube_api.get_video_transcript(video_id)
        if transcript:
            return jsonify({"transcript": transcript})
        else:
            return jsonify({"transcript": None}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Additional API endpoints for data visualization can be added here

if __name__ == "__main__":
    app.run(debug=True)
