import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import websockets
from wordcloud import WordCloud
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import json


def visualize_sentiment_trends(sentiment_data, output_file):
    """
    Visualize sentiment trends over time.
    """
    sentiment_data = handle_large_volumes_of_data(sentiment_data)
    sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])
    fig = px.line(
        sentiment_data, x="date", y="sentiment", title="Sentiment Trends Over Time"
    )
    fig.update_traces(hoverinfo="text+name", mode="lines+markers+text")
    fig.write_html(output_file)


def save_visualization_as_graph(data, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(data["date"], data["sentiment"], marker="o")
    plt.title("Sentiment Analysis")
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def create_interactive_visualization(data, filename):
    fig = px.line(data, x="date", y="sentiment", title="Sentiment Analysis")
    fig.write_html(filename)


async def real_time_visualization(websocket, path):
    async for message in websocket:
        data = pd.read_json(message)
        fig = go.Figure(
            data=[go.Scatter(x=data["date"], y=data["sentiment"], mode="lines+markers")]
        )
        fig.update_layout(
            title="Real-Time Sentiment Analysis",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
        )
        await websocket.send(fig.to_json())


def start_websocket_server():
    start_server = websockets.serve(real_time_visualization, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


def filter_data(data, criteria):
    filtered_data = data
    for key, value in criteria.items():
        filtered_data = filtered_data[filtered_data[key] == value]
    return filtered_data


def visualize_filtered_data(data, criteria, output_file):
    filtered_data = filter_data(data, criteria)
    visualize_sentiment_trends(filtered_data, output_file)


def handle_large_volumes_of_data(data):
    # Implement data aggregation or sampling to handle large datasets
    if len(data) > 10000:
        data = data.sample(n=10000, random_state=42)
    return data


def visualize_large_volumes_of_data(data, output_file):
    handle_large_volumes_of_data(data)
    visualize_sentiment_trends(data, output_file)


def add_tooltips(fig):
    fig.update_traces(hoverinfo="text+name", mode="lines+markers+text")
    return fig


def enable_filtering(data, criteria):
    filtered_data = filter_data(data, criteria)
    return filtered_data


def create_heatmap(data, filename):
    fig = px.density_heatmap(data, x="date", y="sentiment", title="Sentiment Heatmap")
    fig = add_tooltips(fig)
    fig.write_html(filename)


def create_word_cloud(data, filename):
    text = " ".join(data["comment"])
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud")
    plt.savefig(filename)
    plt.close()


def create_sentiment_distribution_chart(data, filename):
    fig = px.histogram(data, x="sentiment", title="Sentiment Distribution")
    fig = add_tooltips(fig)
    fig.write_html(filename)


def visualize_sentiment_by_type(sentiment_data, output_file):
    comments_data = sentiment_data[sentiment_data["input_type"] == "comment"]
    transcripts_data = sentiment_data[sentiment_data["input_type"] == "transcript"]

    if not comments_data.empty:
        fig_comments = px.line(
            comments_data,
            x=comments_data.index,
            y="vader_sentiment",
            title="Comment Sentiment Trends",
        )
        fig_comments = add_tooltips(fig_comments)
        fig_comments.write_html(f"{output_file}_comments.html")

    if not transcripts_data.empty:
        fig_transcripts = px.line(
            transcripts_data,
            x=transcripts_data.index,
            y="vader_sentiment",
            title="Transcript Sentiment Trends",
        )
        fig_transcripts = add_tooltips(fig_transcripts)
        fig_transcripts.write_html(f"{output_file}_transcripts.html")


def visualize_opinion_predictions(opinion_data, output_file):
    fig = px.bar(
        opinion_data,
        x="predicted_opinion",
        title="Distribution of Predicted Opinions",
    )
    fig = add_tooltips(fig)
    fig.write_html(output_file)


def visualize_bias_detection(bias_data, output_file):
    fig = px.bar(bias_data, x="bias_type", title="Types of Bias Detected")
    fig = add_tooltips(fig)
    fig.write_html(output_file)


def visualize_social_issues(social_issue_data, output_file):
    fig = px.bar(
        social_issue_data, x="social_issue", title="Prevalence of Social Issues"
    )
    fig = add_tooltips(fig)
    fig.write_html(output_file)


def visualize_psychological_aspects(psychological_data, output_file):
    fig = px.line(
        psychological_data,
        x=psychological_data.index,
        y="emotional_pattern",
        title="Emotional Patterns",
    )
    fig = add_tooltips(fig)
    fig.write_html(output_file)


def visualize_philosophical_aspects(philosophical_data, output_file):
    fig = px.bar(
        philosophical_data,
        x="underlying_value",
        title="Distribution of Underlying Values and Beliefs",
    )
    fig = add_tooltips(fig)
    fig.write_html(output_file)


def visualize_truth_and_objectivity(truth_data, output_file):
    fig = px.bar(truth_data, x="source_credibility", title="Credibility of Sources")
    fig = add_tooltips(fig)
    fig.write_html(output_file)


def create_sentiment_timeline(sentiment_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive timeline of sentiment scores.
    """
    df = pd.DataFrame(sentiment_data)
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"])
        if "timestamp" in df.columns
        else pd.date_range(end="now", periods=len(df))
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["polarity"],
            mode="lines+markers",
            name="Sentiment Score",
            hovertemplate="%{y:.2f}",
        )
    )

    fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        template="plotly_dark",
    )

    return fig


def create_sentiment_distribution(sentiment_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a distribution plot of sentiment scores.
    """
    df = pd.DataFrame(sentiment_data)

    fig = px.histogram(
        df,
        x="polarity",
        nbins=50,
        title="Distribution of Sentiment Scores",
        template="plotly_dark",
    )

    fig.update_layout(xaxis_title="Sentiment Score", yaxis_title="Count")

    return fig


def create_sentiment_summary(sentiment_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a summary dashboard of sentiment analysis.
    """
    df = pd.DataFrame(sentiment_data)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Sentiment Distribution",
            "Sentiment vs. Subjectivity",
            "Top Keywords",
            "Sentiment by Category",
        ),
    )

    # Sentiment Distribution
    fig.add_trace(go.Histogram(x=df["polarity"], name="Sentiment"), row=1, col=1)

    # Sentiment vs Subjectivity Scatter
    fig.add_trace(
        go.Scatter(
            x=df["polarity"], y=df["subjectivity"], mode="markers", name="Comments"
        ),
        row=1,
        col=2,
    )

    # Top Keywords (if available)
    if "keywords" in df.columns:
        keywords = pd.Series(" ".join(df["keywords"]).split()).value_counts()[:10]
        fig.add_trace(
            go.Bar(
                x=keywords.values, y=keywords.index, orientation="h", name="Keywords"
            ),
            row=2,
            col=1,
        )

    # Sentiment by Category
    if "category" in df.columns:
        category_sentiment = df.groupby("category")["polarity"].mean().sort_values()
        fig.add_trace(
            go.Bar(
                x=category_sentiment.index,
                y=category_sentiment.values,
                name="Category Sentiment",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=800, title_text="Sentiment Analysis Dashboard", template="plotly_dark"
    )

    return fig


def create_topic_visualization(topic_data: Dict[str, List[str]]) -> go.Figure:
    """
    Create visualization for topic analysis.
    """
    topics = []
    counts = []

    for topic, texts in topic_data.items():
        topics.append(topic)
        counts.append(len(texts))

    fig = px.bar(
        x=topics, y=counts, title="Topics Distribution", template="plotly_dark"
    )

    fig.update_layout(xaxis_title="Topic", yaxis_title="Count")

    return fig


def save_visualization(fig: go.Figure, filename: str):
    """
    Save visualization to HTML file.
    """
    fig.write_html(f"static/visualizations/{filename}.html")


def generate_report(
    sentiment_data: List[Dict[str, Any]], topic_data: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Generate a comprehensive analysis report.
    """
    df = pd.DataFrame(sentiment_data)

    report = {
        "total_comments": len(df),
        "average_sentiment": df["polarity"].mean(),
        "sentiment_std": df["polarity"].std(),
        "sentiment_distribution": {
            "positive": len(df[df["polarity"] > 0]),
            "neutral": len(df[df["polarity"] == 0]),
            "negative": len(df[df["polarity"] < 0]),
        },
        "topic_distribution": {
            topic: len(texts) for topic, texts in topic_data.items()
        },
        "visualizations": {
            "sentiment_timeline": "sentiment_timeline.html",
            "sentiment_distribution": "sentiment_distribution.html",
            "topic_distribution": "topic_distribution.html",
            "sentiment_summary": "sentiment_summary.html",
        },
    }

    return report


def export_visualizations(data: Dict[str, Any], format: str = "html"):
    """
    Export all visualizations in the specified format.
    """
    sentiment_data = data.get("sentiment_results", [])
    topic_data = data.get("topic_results", {})

    # Create visualizations
    timeline = create_sentiment_timeline(sentiment_data)
    distribution = create_sentiment_distribution(sentiment_data)
    topics = create_topic_visualization(topic_data)
    summary = create_sentiment_summary(sentiment_data)

    # Save visualizations
    if format == "html":
        timeline.write_html("static/visualizations/sentiment_timeline.html")
        distribution.write_html("static/visualizations/sentiment_distribution.html")
        topics.write_html("static/visualizations/topic_distribution.html")
        summary.write_html("static/visualizations/sentiment_summary.html")
    elif format == "json":
        return {
            "timeline": timeline.to_json(),
            "distribution": distribution.to_json(),
            "topics": topics.to_json(),
            "summary": summary.to_json(),
        }
    else:
        raise ValueError("Unsupported export format")
