import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import (
    pipeline,
    BertTokenizer,
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from spellchecker import SpellChecker
import json


def load_config():
    with open("config.json", "r") as f:
        return json.load(f)


# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize HuggingFace sentiment analysis pipeline
hf_analyzer = pipeline("sentiment-analysis")

# Initialize BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# Initialize SentenceTransformer model for contextual embeddings
sentence_model = SentenceTransformer("bert-base-nli-mean-tokens")

# Initialize NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize spell checker
spell = SpellChecker()

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Initialize toxicity detection model and tokenizer
toxic_model_name = "unitary/toxic-bert"
toxic_tokenizer = AutoTokenizer.from_pretrained(toxic_model_name)
toxic_model = AutoModelForSequenceClassification.from_pretrained(toxic_model_name)


def preprocess_comment(comment):
    # Lowercase the comment
    comment = comment.lower()

    # Remove special characters and punctuation
    comment = re.sub(r"[^a-zA-Z\s]", "", comment)

    # Tokenize the comment
    tokens = comment.split()

    # Remove stop-words
    tokens = [word for word in tokens if word not in stopwords.words("english")]

    # Correct spelling
    tokens = [spell.correction(word) for word in tokens]

    # Lemmatize and stem the tokens
    tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens]

    # Handle negations
    tokens = [
        "not_" + word if word in ["not", "never", "no"] else word for word in tokens
    ]

    return " ".join(tokens)


def perform_sentiment_analysis(text_inputs, language="en"):
    config = load_config()
    sentiment_model = config.get("sentiment_model", "vader")
    results = []
    for text in text_inputs:
        if isinstance(text, dict) and "text" in text:
            input_text = text["text"]
            input_type = text.get("type", "comment")
        else:
            input_text = text
            input_type = "comment"

        preprocessed_text = preprocess_comment(input_text)

        if input_type == "transcript":
            topic_results = categorize_comments_by_themes(
                [input_text], text_type="transcript"
            )
        else:
            topic_results = categorize_comments_by_themes(
                [input_text], text_type="comment"
            )

        if sentiment_model == "vader":
            vader_result = vader_analyzer.polarity_scores(preprocessed_text)
            results.append(
                {
                    "input_text": input_text,
                    "input_type": input_type,
                    "preprocessed_text": preprocessed_text,
                    "vader_sentiment": vader_result,
                    "topic_results": topic_results,
                }
            )
        elif sentiment_model == "hf":
            # Use a language-specific sentiment analysis model if the language is not English
            if language != "en":
                try:
                    hf_analyzer_lang = pipeline(
                        "sentiment-analysis",
                        model=f"nlptown/bert-base-multilingual-uncased-sentiment",
                    )
                    hf_result = hf_analyzer_lang(preprocessed_text)[0]
                except Exception as e:
                    hf_result = hf_analyzer(preprocessed_text)[0]
            else:
                hf_result = hf_analyzer(preprocessed_text)[0]
            results.append(
                {
                    "input_text": input_text,
                    "input_type": input_type,
                    "preprocessed_text": preprocessed_text,
                    "hf_sentiment": hf_result,
                    "topic_results": topic_results,
                }
            )
        elif sentiment_model == "bert":
            # BERT sentiment analysis
            inputs = bert_tokenizer(
                preprocessed_text, return_tensors="pt", truncation=True, padding=True
            )
            outputs = bert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            bert_result = {
                "label": "positive" if probs[0][1] > probs[0][0] else "negative",
                "score": (
                    probs[0][1].item()
                    if probs[0][1] > probs[0][0]
                    else probs[0][0].item()
                ),
            }
            results.append(
                {
                    "input_text": input_text,
                    "input_type": input_type,
                    "preprocessed_text": preprocessed_text,
                    "bert_sentiment": bert_result,
                    "topic_results": topic_results,
                }
            )
    return pd.DataFrame(results)


def categorize_comments_by_themes(texts, text_type="comment"):
    # Preprocess texts
    preprocessed_texts = [preprocess_comment(text) for text in texts]

    # Convert texts to contextual embeddings
    embeddings = sentence_model.encode(preprocessed_texts)

    # Topic modeling using LDA
    vectorizer = CountVectorizer()
    vectorized_texts = vectorizer.fit_transform(preprocessed_texts)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(vectorized_texts)
    lda_topics = []
    for topic_idx, topic in enumerate(lda.components_):
        lda_topics.append(
            {
                "topic_id": topic_idx,
                "words": [
                    vectorizer.get_feature_names_out()[i]
                    for i in topic.argsort()[:-11:-1]
                ],
            }
        )

    # Topic modeling using NMF
    nmf = NMF(n_components=5, random_state=42)
    nmf.fit(vectorized_texts)
    nmf_topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        nmf_topics.append(
            {
                "topic_id": topic_idx,
                "words": [
                    vectorizer.get_feature_names_out()[i]
                    for i in topic.argsort()[:-11:-1]
                ],
            }
        )

    # Hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=5)
    clusters = clustering.fit_predict(embeddings)

    return {
        "lda_topics": lda_topics,
        "nmf_topics": nmf_topics,
        "clusters": clusters,
        "text_type": text_type,
    }


def incorporate_user_feedback(feedback_data, sentiment_data):
    for feedback in feedback_data:
        comment_id = feedback["comment_id"]
        corrected_sentiment = feedback["corrected_sentiment"]
        sentiment_data.loc[
            sentiment_data["comment_id"] == comment_id, "corrected_sentiment"
        ] = corrected_sentiment
    sentiment_data.to_csv("sentiment_data.csv", index=False)
    return sentiment_data


def calculate_sentiment_intensity(sentiment_data):
    sentiment_data["sentiment_intensity"] = sentiment_data.apply(
        lambda row: row["vader_sentiment"]["compound"], axis=1
    )
    return sentiment_data


def calculate_sentiment_variance(sentiment_data):
    sentiment_data["sentiment_variance"] = (
        sentiment_data["sentiment_intensity"].rolling(window=10).var()
    )
    return sentiment_data


def calculate_sentiment_correlation(sentiment_data, other_data):
    sentiment_data["sentiment_correlation"] = (
        sentiment_data["sentiment_intensity"].rolling(window=10).corr(other_data)
    )
    return sentiment_data


def calculate_sentiment_clustering(sentiment_data):
    embeddings = sentence_model.encode(sentiment_data["preprocessed_comment"].tolist())
    clustering = AgglomerativeClustering(n_clusters=5)
    sentiment_data["sentiment_cluster"] = clustering.fit_predict(embeddings)
    return sentiment_data


def calculate_sentiment_propagation(sentiment_data):
    sentiment_data["sentiment_propagation"] = (
        sentiment_data["sentiment_intensity"].diff().abs()
    )
    return sentiment_data


def calculate_sentiment_echo_chambers(sentiment_data):
    sentiment_data["sentiment_echo_chamber"] = (
        sentiment_data.groupby(
            sentiment_data["sentiment_cluster"]
            .rolling(window=10)
            .apply(lambda x: tuple(x))
        )["vader_sentiment"]
        .apply(lambda x: len(set([i["compound"] > 0 for i in x])) == 1)
        .reset_index(drop=True)
    )
    return sentiment_data


def calculate_sentiment_shifts(sentiment_data):
    sentiment_data["sentiment_shift"] = sentiment_data["sentiment_intensity"].diff()
    return sentiment_data


def calculate_sentiment_spikes(sentiment_data):
    sentiment_data["sentiment_spike"] = (
        sentiment_data["sentiment_intensity"].diff().abs() > 0.5
    )
    return sentiment_data


def calculate_sentiment_engagement(sentiment_data, engagement_data):
    sentiment_data["sentiment_engagement"] = engagement_data
    return sentiment_data


def calculate_sentiment_influence(sentiment_data, influence_data):
    sentiment_data["sentiment_influence"] = influence_data
    return sentiment_data


def generate_dynamic_suggestions(sentiment_data):
    suggestions = []
    if sentiment_data.empty:
        return suggestions

    # Example suggestion: Identify comments with high negative sentiment
    negative_comments = sentiment_data[
        sentiment_data["vader_sentiment"].apply(lambda x: x["compound"] < -0.5)
    ]
    if not negative_comments.empty:
        suggestions.append(
            f"Identified {len(negative_comments)} comments with high negative sentiment."
        )

    # Example suggestion: Identify comments with high positive sentiment
    positive_comments = sentiment_data[
        sentiment_data["vader_sentiment"].apply(lambda x: x["compound"] > 0.5)
    ]
    if not positive_comments.empty:
        suggestions.append(
            f"Identified {len(positive_comments)} comments with high positive sentiment."
        )

    # Example suggestion: Identify comments with neutral sentiment
    neutral_comments = sentiment_data[
        sentiment_data["vader_sentiment"].apply(lambda x: abs(x["compound"]) <= 0.1)
    ]
    if not neutral_comments.empty:
        suggestions.append(
            f"Identified {len(neutral_comments)} comments with neutral sentiment."
        )

    # Example suggestion: Identify comments with high variance in sentiment
    if "sentiment_variance" in sentiment_data.columns:
        high_variance_comments = sentiment_data[
            sentiment_data["sentiment_variance"] > 0.2
        ]
        if not high_variance_comments.empty:
            suggestions.append(
                f"Identified {len(high_variance_comments)} comments with high variance in sentiment."
            )

    # Example suggestion: Identify comments with sentiment shifts
    if "sentiment_shift" in sentiment_data.columns:
        significant_shifts = sentiment_data[
            sentiment_data["sentiment_shift"].abs() > 0.3
        ]
        if not significant_shifts.empty:
            suggestions.append(
                f"Identified {len(significant_shifts)} comments with significant sentiment shifts."
            )


# Example suggestion: Identify potential manipulation or bias
if (
    "sentiment_shift" in sentiment_data.columns
    and "sentiment_echo_chamber" in sentiment_data.columns
):
    biased_comments = sentiment_data[
        (sentiment_data["sentiment_shift"].abs() > 0.3)
        & (sentiment_data["sentiment_echo_chamber"] == True)
    ]
    if not biased_comments.empty:
        suggestions.append(
            f"Identified {len(biased_comments)} comments with significant sentiment shifts within echo chambers, which may indicate potential manipulation or bias."
        )

# Example suggestion: Identify potential echo chambers
if "sentiment_echo_chamber" in sentiment_data.columns:
    echo_chamber_comments = sentiment_data[
        sentiment_data["sentiment_echo_chamber"] == True
    ]
    if not echo_chamber_comments.empty:
        suggestions.append(
            f"Identified {len(echo_chamber_comments)} comments within potential echo chambers."
        )

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

    def predict_opinions(self, texts, text_type="comment", language="en"):
        # Placeholder for opinion prediction logic
        # This will use machine learning models to predict user opinions
        # based on comment sentiment, keywords, and other features.
        return {"opinion_predictions": "Opinion predictions not yet implemented"}

    def detect_bias(self, texts, text_type="comment", language="en"):
        # Placeholder for bias detection logic
        # This will use NLP techniques to detect bias in comments and video content.
        return {"bias_detection_results": "Bias detection not yet implemented"}

    def analyze_social_issues(self, texts, text_type="comment", language="en"):
        # Placeholder for social issue analysis logic
        # This will identify and analyze social issues discussed in comments and video content.
        return {"social_issue_analysis": "Social issue analysis not yet implemented"}

    def analyze_psychological_aspects(self, texts, text_type="comment", language="en"):
        # Placeholder for psychological analysis logic
        # This will analyze the psychological aspects of comments and video content.
        return {"psychological_analysis": "Psychological analysis not yet implemented"}

    def analyze_philosophical_aspects(self, texts, text_type="comment", language="en"):
        # Placeholder for philosophical analysis logic
        # This will analyze the philosophical aspects of comments and video content.
        return {"philosophical_analysis": "Philosophical analysis not yet implemented"}

    def analyze_truth_and_objectivity(self, texts, text_type="comment", language="en"):
        # Placeholder for truth and objectivity analysis logic
        # This will analyze the truth and objectivity of video content and comments.
        return {
            "truth_and_objectivity_analysis": "Truth and objectivity analysis not yet implemented"
        }
