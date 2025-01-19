import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import websockets
from wordcloud import WordCloud


def visualize_sentiment_trends(sentiment_data, output_file):
    sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])
    fig = px.line(
        sentiment_data, x="date", y="sentiment", title="Sentiment Trends Over Time"
    )
    fig = add_tooltips(fig)
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


import cachetools

cache = cachetools.TTLCache(maxsize=100, ttl=3600)  # 1 hour cache


@cachetools.cached(cache)
def handle_large_volumes_of_data(data):
    """Handles large volumes of data by downsampling."""
    if len(data) > 10000:
        logging.info("Downsampling data for visualization.")
        data = data.sample(frac=0.1)  # Downsample to 10%
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


def visualize_sentiment_trends(sentiment_data, output_file):
    sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])
    fig = px.line(
        sentiment_data, x="date", y="sentiment", title="Sentiment Trends Over Time"
    )
    fig.update_traces(hoverinfo="text+name", mode="lines+markers+text")
    fig.write_html(output_file)


def sentiment_trend_analysis(sentiment_data, output_file):
    sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])
    fig = px.line(
        sentiment_data, x="date", y="sentiment", title="Sentiment Trend Analysis"
    )
    fig.update_traces(hoverinfo="text+name", mode="lines+markers+text")
    fig.write_html(output_file)


def sentiment_distribution_analysis(sentiment_data, output_file):
    fig = px.histogram(
        sentiment_data, x="sentiment", title="Sentiment Distribution Analysis"
    )
    fig.write_html(output_file)


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
    fig.write_html(output_file)a

from plotly.subplots import make_subplots


def create_advanced_sentiment_visualization(data, output_file):
    """Create advanced interactive sentiment visualization with multiple views."""
    try:
        # Handle empty data
        if data.empty:
            raise ValueError("No data available for visualization")

        # Create subplots with error handling
        try:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Sentiment Over Time",
                    "Comment Themes",
                    "Sentiment Distribution",
                    "Topic Evolution",
                ),
            )
        except Exception as e:
            print(f"Error creating subplots: {e}")
            return None

        # Add sentiment timeline with error handling
        try:
            fig.add_trace(
                go.Scatter(
                    x=data["date"],
                    y=data["sentiment"],
                    mode="lines+markers",
                    name="Sentiment",
                ),
                row=1,
                col=1,
            )
        except Exception as e:
            print(f"Error adding sentiment timeline: {e}")

        # Save visualization
        try:
            fig.write_html(output_file)
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return None

    except Exception as e:
        print(f"Error in visualization creation: {e}")
        return None


def detect_toxic_comments(self, text_inputs):
    results = []
    for text in text_inputs:
        if isinstance(text, dict) and "text" in text:
            input_text = text["text"]
            input_type = text.get("type", "comment")
        else:
            input_text = text
            input_type = "comment"

        inputs = toxic_tokenizer(
            input_text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = toxic_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        toxic_score = probs[0][1].item()

        results.append(
            {
                "input_text": input_text,
                "input_type": input_type,
                "toxic_score": toxic_score,
                "is_toxic": toxic_score > 0.5,
            }
        )
    return pd.DataFrame(results)


def visualize_user_engagement(data, output_file):
    """Create visualization for user engagement patterns."""
    fig = go.Figure()

    # Add engagement metrics
    fig.add_trace(
        go.Scatter(x=data["date"], y=data["likes"], name="Likes", mode="lines+markers")
    )

    fig.add_trace(
        go.Scatter(
            x=data["date"], y=data["replies"], name="Replies", mode="lines+markers"
        )
    )

    fig.update_layout(
        title="User Engagement Over Time",
        xaxis_title="Date",
        yaxis_title="Count",
        hovermode="x unified",
    )

    fig.write_html(output_file)


def create_topic_network(data, output_file):
    """Create interactive network visualization of related topics."""
    # Network visualization code here
    pass
