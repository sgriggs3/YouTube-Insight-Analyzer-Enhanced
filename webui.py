import os
from flask import Flask, render_template, request, jsonify, make_response
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
from flask_compress import Compress
from flask_cors import CORS
import msgpack
import uuid
import logging
from functools import wraps

app = Flask(__name__)


# Add compression
def init_app(app):
    Compress(app)
    CORS(app)
    app.config["COMPRESS_ALGORITHM"] = ["br", "gzip"]
    app.config["COMPRESS_LEVEL"] = 6
    app.config["COMPRESS_MIN_SIZE"] = 500


init_app(app)


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


config = load_config()
web_ui_port = config.get("web_ui_port", 8080)

# Load data
data = pd.read_csv("sentiment_data.csv")


# Add performance middleware
@app.before_request
def before_request():
    # Add cache headers for static assets
    if request.path.startswith("/static/"):
        return make_response(
            render_template("static.html"),
            200,
            {"Cache-Control": "public, max-age=31536000", "Vary": "Accept-Encoding"},
        )


# Add logging configuration 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add input validation helper
def validate_url(url):
    if not url or not isinstance(url, str):
        raise ValueError("Invalid URL provided")
    # Add more validation as needed


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
    async def connect_websocket_with_retry():
            [
                comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for comment in comments
            ]
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


async def process_message(data, connection_id):
    if data.type == "start_analysis":
        urls = data.urls
        options = data.options
        results = []
        for i, url in enumerate(urls):
            try:
                video_id = youtube_api.extract_video_id(url)
                comments = youtube_api.get_video_comments(video_id)
                sentiment_results = sentiment_analysis.perform_sentiment_analysis(
                    [
                        comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                        for comment in comments
                    ]
                )
                results.append(sentiment_results)
                progress = (i + 1) / len(urls) * 100
                await send_progress_update(connection_id, progress)
            except Exception as e:
                results.append({"error": f"Error processing URL: {url}, {str(e)}"})
                await send_progress_update(
                    connection_id, 0, f"Error processing URL: {url}, {str(e)}"
                )
        await send_results(connection_id, results)


async def send_progress_update(connection_id, progress, error=None):
    if connection_id in CONNECTIONS:
        websocket = CONNECTIONS[connection_id]
        message = {"type": "progress_update", "progress": progress, "error": error}
        await websocket.send(json.dumps(message))


async def send_results(connection_id, results):
    if connection_id in CONNECTIONS:
        websocket = CONNECTIONS[connection_id]
        message = {"type": "analysis_results", "results": results}
        await websocket.send(json.dumps(message))


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
            [
                comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for comment in comments
            ]
        )
        youtube_api.save_data_to_csv(
            sentiment_results, f"{video_id}_sentiment_analysis.csv"
        )
        return jsonify({"message": "URL processed successfully", "video_id": video_id})
    return render_template("input_url_form.html")


# Add request validation decorator
def validate_request(*required_fields):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400
            
            for field in required_fields:
                if field not in request.json:
                    return jsonify({"error": f"Missing required field: {field}"}), 400
                    
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route("/start-sentiment-analysis", methods=["POST"])
@validate_request("urls", "scrape_type", "comment_limit")
def start_sentiment_analysis():
    try:
        urls = request.json["urls"]
        scrape_type = request.json["scrape_type"]
        comment_limit = request.json["comment_limit"]
        
        logger.info(f"Starting analysis of {len(urls)} URLs with type {scrape_type} and limit {comment_limit}")
        results = []
        
        for url in urls:
            try:
                video_id = youtube_api.extract_video_id(url)
                if not video_id:
                    raise ValueError(f"Invalid YouTube URL: {url}")
                    
                logger.info(f"Processing video {video_id}")
                comments = youtube_api.get_video_comments(video_id, scrape_type, comment_limit)
                sentiment_results = sentiment_analysis.perform_sentiment_analysis(comments)
                results.append(sentiment_results)
                
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                results.append({"error": str(e), "url": url})
                
        return jsonify({
            "message": "Analysis complete",
            "results": results
        })
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return jsonify({
            "error": "Analysis failed", 
            "message": str(e)
        }), 400

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


@app.route("/review-feedback", methods=["POST"])
def review_feedback():
    feedback = request.json.get("feedback")
    sentiment_data = pd.read_csv("sentiment_data.csv")
    updated_sentiment_data = sentiment_analysis.review_and_refine_feedback(
        feedback, sentiment_data
    )
    updated_sentiment_data.to_csv("sentiment_data.csv", index=False)
    return jsonify({"message": "Feedback reviewed and refined successfully"})


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
                    "text": comment["snippet"]["topLevelComment"]["snippet"][
                        "textDisplay"
                    ],
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
    return jsonify(sentiment_results.to_dict(orient="records"))


@app.route("/api/visualizations/<video_id>", methods=["GET"])
def api_visualizations(video_id):
    comments = youtube_api.get_video_comments(video_id)
    transcript = youtube_api.get_video_transcript(video_id)

    text_inputs = []
    if comments:
        text_inputs.extend(
            [
                {
                    "text": comment["snippet"]["topLevelComment"]["snippet"][
                        "textDisplay"
                    ],
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


@app.route("/api/analyze/<video_id>", methods=["POST"])
@validate_request("analysis_type")
def analyze_video(video_id):
    try:
        analysis_type = request.json["analysis_type"]
        
        # Get video metadata and comments
        metadata = youtube_api.get_video_metadata(video_id)
        comments = youtube_api.get_video_comments(video_id)
        
        results = {}
        
        if analysis_type == "sentiment":
            sentiment_data = sentiment_analysis.perform_sentiment_analysis(comments)
            results["sentiment"] = sentiment_data
            
        elif analysis_type == "engagement":
            engagement_data = data_visualization.analyze_engagement(comments, metadata)
            results["engagement"] = engagement_data
            
        elif analysis_type == "topics":
            topic_data = sentiment_analysis.analyze_topics(comments)
            results["topics"] = topic_data
            
        else:
            return jsonify({"error": "Invalid analysis type"}), 400
            
        return jsonify({
            "success": True,
            "video_id": video_id,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            "error": "Analysis failed",
            "message": str(e)
        }), 500


@app.route("/start-scraping", methods=["POST"])
@validate_request("url", "scrape_type", "comment_limit")
def start_scraping():
    try:
        url = request.json["url"]
        scrape_type = request.json["scrape_type"]
        comment_limit = request.json["comment_limit"]
        
        video_id = youtube_api.extract_video_id(url)
        if not video_id:
            raise ValueError(f"Invalid YouTube URL: {url}")
        
        logger.info(f"Starting scraping for video {video_id} with type {scrape_type} and limit {comment_limit}")
        
        # Start scraping in a separate thread to avoid blocking
        threading.Thread(target=scrape_comments, args=(video_id, scrape_type, comment_limit)).start()
        
        return jsonify({"message": "Scraping started successfully", "video_id": video_id})
    
    except Exception as e:
        logger.error(f"Error starting scraping: {str(e)}")
        return jsonify({"error": "Scraping failed", "message": str(e)}), 500

def scrape_comments(video_id, scrape_type, comment_limit):
    try:
        comments = youtube_api.get_video_comments(video_id, scrape_type, comment_limit)
        youtube_api.save_data_to_csv(comments, f"{video_id}_comments.csv")
        logger.info(f"Scraping completed for video {video_id}")
        
        # Notify clients about the completion
        asyncio.run(broadcast_scraping_progress(video_id, 100, "Scraping completed"))
        
    except Exception as e:
        logger.error(f"Error scraping comments for video {video_id}: {str(e)}")
        asyncio.run(broadcast_scraping_progress(video_id, 0, f"Error: {str(e)}"))

async def broadcast_scraping_progress(video_id, progress, message):
    if video_id in CONNECTIONS:
        websocket = CONNECTIONS[video_id]
        await websocket.send(json.dumps({"type": "scraping_progress", "progress": progress, "message": message}))

import threading

CONNECTIONS = {}
MAX_RETRIES = 3


async def optimized_ws_handler(websocket, path):
    connection_id = str(uuid.uuid4())
    CONNECTIONS[connection_id] = websocket
    retry_count = 0

    try:
        async for message in websocket:
            try:
                # Handle binary messages efficiently
                if isinstance(message, bytes):
                    data = msgpack.unpackb(message)
                else:
                    data = json.loads(message)

                # Add timeout for processing
                async with asyncio.timeout(30):
                    await process_message(data, connection_id)

                retry_count = 0  # Reset on successful processing

            except asyncio.TimeoutError:
                logging.error(f"Message processing timeout for {connection_id}")
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                    continue
                break

    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        del CONNECTIONS[connection_id]
        await websocket.close()


async def main():
    start_server = websockets.serve(optimized_ws_handler, "localhost", 8765)
    flask_thread = threading.Thread(
        target=app.run,
        kwargs={"debug": True, "port": web_ui_port, "use_reloader": False},
    )
    flask_thread.start()
    await start_server


if __name__ == "__main__":
    asyncio.run(main())

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "message": str(error)
    }), 500
