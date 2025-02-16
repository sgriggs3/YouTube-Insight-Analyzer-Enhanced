import pandas as pd
import numpy as np
from data_visualization import (
    visualize_sentiment_trends,
    create_interactive_visualization,
    create_heatmap,
    create_word_cloud,
    create_sentiment_distribution_chart,
    visualize_sentiment_by_type,
    visualize_opinion_predictions,
    visualize_bias_detection,
    visualize_social_issues,
    visualize_psychological_aspects,
    visualize_philosophical_aspects,
    visualize_truth_and_objectivity,
    create_advanced_sentiment_visualization,
    visualize_user_engagement,
)

# Sample data
dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
sentiment_scores = np.random.uniform(-1, 1, 10)
comments = [
    "This is a great comment",
    "I am very disappointed",
    "The weather is okay",
    "This is fantastic",
    "The transcript was informative",
    "I love this",
    "This is terrible",
    "I am neutral",
    "This is amazing",
    "This is bad",
]
input_types = [
    "comment",
    "comment",
    "comment",
    "comment",
    "transcript",
    "comment",
    "comment",
    "comment",
    "comment",
    "comment",
]
likes = np.random.randint(0, 100, 10)
replies = np.random.randint(0, 50, 10)

data = pd.DataFrame(
    {
        "date": dates,
        "sentiment": sentiment_scores,
        "comment": comments,
        "input_type": input_types,
        "likes": likes,
        "replies": replies,
    }
)

opinion_data = pd.DataFrame(
    {"predicted_opinion": ["positive", "negative", "neutral", "positive", "negative"]}
)

bias_data = pd.DataFrame(
    {"bias_type": ["gender", "race", "political", "gender", "race"]}
)

social_issue_data = pd.DataFrame(
    {"social_issue": ["poverty", "inequality", "racism", "poverty", "inequality"]}
)

psychological_data = pd.DataFrame({"emotional_pattern": np.random.uniform(-1, 1, 10)})

philosophical_data = pd.DataFrame(
    {"underlying_value": ["freedom", "justice", "equality", "freedom", "justice"]}
)

truth_data = pd.DataFrame(
    {"source_credibility": ["high", "low", "medium", "high", "low"]}
)

# Create visualizations
visualize_sentiment_trends(data, "sentiment_trends.html")
create_interactive_visualization(data, "interactive_visualization.html")
create_heatmap(data, "heatmap.html")
create_word_cloud(data, "wordcloud.png")
create_sentiment_distribution_chart(data, "sentiment_distribution.html")
visualize_sentiment_by_type(data, "sentiment_by_type")
visualize_opinion_predictions(opinion_data, "opinion_predictions.html")
visualize_bias_detection(bias_data, "bias_detection.html")
visualize_social_issues(social_issue_data, "social_issues.html")
visualize_psychological_aspects(psychological_data, "psychological_aspects.html")
visualize_philosophical_aspects(philosophical_data, "philosophical_aspects.html")
visualize_truth_and_objectivity(truth_data, "truth_and_objectivity.html")
create_advanced_sentiment_visualization(data, "advanced_sentiment_visualization.html")
visualize_user_engagement(data, "user_engagement.html")
