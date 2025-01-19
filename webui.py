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

app = Flask(__name__)

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
        comments = youtube_api.get_video_comments(video_id, api_key)
        sentiment_results = sentiment_analysis.perform_sentiment_analysis(
            [
                comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for comment in comments
            ]
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


@app.route("/scraping")
def scraping():
    return render_template("scraping.html")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/examples")
def examples():
    return render_template("examples.html")


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
        comments = youtube_api.get_video_comments(video_id, api_key)
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


@app.route("/start-sentiment-analysis", methods=["POST"])
def start_sentiment_analysis():
    urls = request.json.get("urls")
    options = request.json.get("options")
    results = []
    for url in urls:
        video_id = youtube_api.extract_video_id(url)
        comments = youtube_api.get_video_comments(video_id, api_key)
        sentiment_results = sentiment_analysis.perform_sentiment_analysis(
            [
                comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for comment in comments
            ]
        )
        results.append(sentiment_results)
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


if __name__ == "__main__":
    start_websocket_server()
    app.run(debug=True)
