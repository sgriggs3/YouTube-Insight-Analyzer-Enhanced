of code and executing then create log file called "tasks-finished" where you give insurctions and srtailsimport pandadetails and on the task and what was fixed or improved. Continue saving the progress to that file.
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
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any
import logging


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
            result = {
                "text": text,
                "sentiment": "positive" if sentiment.polarity > 0 else "negative",
                "score": abs(sentiment.polarity),
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity,
            }
            results.append(result)

    return results


def categorize_comments_by_themes(
    texts: List[str], text_type: str = "comment"
) -> Dict[str, List[str]]:
    """
    Categorize comments into themes using NLP.

    Args:
        texts: List of text strings to analyze
        text_type: Type of text ("comment" or "transcript")

    Returns:
        Dictionary mapping themes to lists of relevant texts
    """
    themes = defaultdict(list)

    # Define key topics/themes to look for
    topic_keywords = {
        "technical": ["quality", "audio", "video", "resolution", "buffer"],
        "content": ["interesting", "boring", "informative", "helpful"],
        "emotional": ["love", "hate", "amazing", "terrible"],
        "critique": ["disagree", "agree", "wrong", "right", "should"],
    }

    for text in texts:
        doc = nlp(text.lower())

        # Get main topics from noun chunks and named entities
        topics = set()
        topics.update([chunk.text for chunk in doc.noun_chunks])
        topics.update([ent.text for ent in doc.ents])

        # Categorize based on keywords
        for theme, keywords in topic_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                themes[theme].append(text)

        # Add general topics category
        if topics:
            themes["topics"].extend(list(topics))

    return dict(themes)


def analyze_sentiment_trends(sentiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze trends in sentiment results over time.

    Args:
        sentiment_results: List of sentiment analysis results

    Returns:
        Dictionary containing trend analysis
    """
    df = pd.DataFrame(sentiment_results)

    trends = {
        "average_sentiment": df["polarity"].mean(),
        "sentiment_std": df["polarity"].std(),
        "subjectivity_mean": df["subjectivity"].mean(),
        "sentiment_counts": df["sentiment"].value_counts().to_dict(),
        "strong_reactions": len(df[df["score"] > 0.9]),  # High confidence sentiments
    }

    return trends


def get_key_phrases(texts: List[str], min_count: int = 2) -> List[str]:
    """Extract key phrases that appear frequently in the texts."""
    phrases = defaultdict(int)

    for text in texts:
        doc = nlp(text)

        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Only phrases
                phrases[chunk.text.lower()] += 1

    # Filter by minimum count
    return [phrase for phrase, count in phrases.items() if count >= min_count]


def export_analysis(
    sentiment_results: List[Dict[str, Any]],
    trends: Dict[str, Any],
    format: str = "json",
) -> str:
    """Export analysis results in specified format."""
    df = pd.DataFrame(sentiment_results)

    if format == "csv":
        return df.to_csv(index=False)
    elif format == "json":
        return {
            "sentiment_results": sentiment_results,
            "trends": trends,
            "summary": {
                "total_analyzed": len(sentiment_results),
                "average_sentiment": trends["average_sentiment"],
                "top_sentiments": trends["sentiment_counts"],
            },
        }
    else:
        raise ValueError("Unsupported export format")


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


def detect_toxic_comments(text_inputs):
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


def predict_opinions(texts, text_type="comment", language="en"):
    """
    Placeholder for opinion prediction logic.
    """
    return {"opinion_predictions": "Opinion predictions not yet implemented"}


def detect_bias(texts, text_type="comment", language="en"):
    """
    Placeholder for bias detection logic.
    """
    return {"bias_detection_results": "Bias detection not yet implemented"}


def analyze_social_issues(texts, text_type="comment", language="en"):
    """
    Placeholder for social issue analysis logic.
    """
    return {"social_issue_analysis": "Social issue analysis not yet implemented"}


def analyze_psychological_aspects(texts, text_type="comment", language="en"):
    """
    Placeholder for psychological analysis logic.
    """
    return {"psychological_analysis": "Psychological analysis not yet implemented"}


def analyze_philosophical_aspects(texts, text_type="comment", language="en"):
    """
    Placeholder for philosophical analysis logic.
    """
    return {"philosophical_analysis": "Philosophical analysis not yet implemented"}


def analyze_truth_and_objectivity(texts, text_type="comment", language="en"):
    """
    Placeholder for truth and objectivity analysis logic.
    """
    return {
        "truth_and_objectivity_analysis": "Truth and objectivity analysis not yet implemented"
    }


class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analysis models"""
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
            )
        except Exception as e:
            logging.error(f"Error initializing sentiment pipeline: {e}")
            self.sentiment_pipeline = None
        # Download required NLTK data
        nltk.download("punkt", quiet=True)
        # Initialize transformers pipeline for more accurate sentiment analysis
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.logger = logging.getLogger(__name__)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text."""
        try:
            if self.sentiment_pipeline:
                result = self.sentiment_pipeline(text)[0]
                return {
                    "text": text,
                    "label": result["label"],
                    "score": result["score"],
                }
            else:
                vader_result = vader_analyzer.polarity_scores(text)
                return {
                    "text": text,
                    "label": (
                        "POSITIVE"
                        if vader_result["compound"] >= 0.05
                        else (
                            "NEGATIVE"
                            if vader_result["compound"] <= -0.05
                            else "NEUTRAL"
                        )
                    ),
                    "score": vader_result["compound"],
                }
        except Exception as e:
            logging.error(f"Error analyzing text: {e}")
            return {"text": text, "label": "ERROR", "score": 0.0}

    def analyze_comments(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of multiple comments."""
        results = []
        overall_sentiment = {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "avg_polarity": 0,
            "avg_subjectivity": 0,
        }

        try:
            texts = [comment["text"] for comment in comments]
            sentiments = self.analyze_texts(texts)
            for sentiment in sentiments:
                results.append(sentiment)
                if sentiment["label"] == "POSITIVE":
                    overall_sentiment["positive"] += 1
                elif sentiment["label"] == "NEGATIVE":
                    overall_sentiment["negative"] += 1
                else:
                    overall_sentiment["neutral"] += 1
                overall_sentiment["avg_polarity"] += sentiment["score"]
            overall_sentiment["avg_polarity"] /= len(sentiments)
        except Exception as e:
            logging.error(f"Error analyzing comments: {e}")

        return {"comments": results, "overall_sentiment": overall_sentiment}

    def categorize_comments_by_themes(
        self, comments: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize comments by common themes/topics."""
        themes = {}

        try:
            # Basic theme categorization - this could be enhanced with topic modeling
            keywords = {
                "technical": [
                    "code",
                    "bug",
                    "error",
                    "fix",
                    "issue",
                    "problem",
                    "solution",
                ],
                "positive_feedback": [
                    "great",
                    "awesome",
                    "amazing",
                    "good",
                    "love",
                    "excellent",
                ],
                "negative_feedback": [
                    "bad",
                    "poor",
                    "terrible",
                    "hate",
                    "awful",
                    "worst",
                ],
                "suggestion": [
                    "suggest",
                    "improvement",
                    "should",
                    "could",
                    "would",
                    "maybe",
                ],
                "question": ["how", "what", "why", "when", "where", "who", "?"],
            }

            for comment in comments:
                text = comment["text"].lower()

                # Categorize comment based on keywords
                for theme, words in keywords.items():
                    if any(word in text for word in words):
                        if theme not in themes:
                            themes[theme] = []
                        themes[theme].append(comment)

            return themes

        except Exception as e:
            self.logger.error(f"Error categorizing comments: {e}")
            raise
