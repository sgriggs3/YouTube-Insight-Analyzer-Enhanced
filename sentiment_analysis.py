import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
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

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Initialize HuggingFace sentiment analysis pipeline
hf_analyzer = pipeline("sentiment-analysis")

# Initialize BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Initialize SentenceTransformer model for contextual embeddings
sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Initialize NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize spell checker
spell = SpellChecker()

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_comment(comment):
    # Lowercase the comment
    comment = comment.lower()
    
    # Remove special characters and punctuation
    comment = re.sub(r'[^a-zA-Z\s]', '', comment)
    
    # Tokenize the comment
    tokens = comment.split()
    
    # Remove stop-words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    # Correct spelling
    tokens = [spell.correction(word) for word in tokens]
    
    # Lemmatize and stem the tokens
    tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in tokens]
    
    # Handle negations
    tokens = ['not_' + word if word in ['not', 'never', 'no'] else word for word in tokens]
    
    return ' '.join(tokens)

def perform_sentiment_analysis(comments):
    results = []
    for comment in comments:
        preprocessed_comment = preprocess_comment(comment)
        
        vader_result = vader_analyzer.polarity_scores(preprocessed_comment)
        hf_result = hf_analyzer(preprocessed_comment)[0]
        
        # BERT sentiment analysis
        inputs = bert_tokenizer(preprocessed_comment, return_tensors='pt', truncation=True, padding=True)
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        bert_result = {'label': 'positive' if probs[0][1] > probs[0][0] else 'negative', 'score': probs[0][1].item() if probs[0][1] > probs[0][0] else probs[0][0].item()}
        
        results.append({
            "comment": comment,
            "preprocessed_comment": preprocessed_comment,
            "vader_sentiment": vader_result,
            "hf_sentiment": hf_result,
            "bert_sentiment": bert_result
        })
    return pd.DataFrame(results)

def categorize_comments_by_themes(comments):
    # Preprocess comments
    preprocessed_comments = [preprocess_comment(comment) for comment in comments]
    
    # Convert comments to contextual embeddings
    embeddings = sentence_model.encode(preprocessed_comments)
    
    # Topic modeling using LDA
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(preprocessed_comments)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_topics = lda.fit_transform(X)
    
    # Topic modeling using NMF
    nmf = NMF(n_components=5, random_state=42)
    nmf_topics = nmf.fit_transform(X)
    
    # Hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=5)
    clusters = clustering.fit_predict(embeddings)
    
    return {
        "lda_topics": lda_topics,
        "nmf_topics": nmf_topics,
        "clusters": clusters
    }

def incorporate_user_feedback(feedback_data, sentiment_data):
    for feedback in feedback_data:
        comment_id = feedback['comment_id']
        corrected_sentiment = feedback['corrected_sentiment']
        sentiment_data.loc[sentiment_data['comment_id'] == comment_id, 'corrected_sentiment'] = corrected_sentiment
    return sentiment_data

def calculate_sentiment_intensity(sentiment_data):
    sentiment_data['sentiment_intensity'] = sentiment_data.apply(lambda row: row['vader_sentiment']['compound'], axis=1)
    return sentiment_data

def calculate_sentiment_variance(sentiment_data):
    sentiment_data['sentiment_variance'] = sentiment_data['sentiment_intensity'].rolling(window=10).var()
    return sentiment_data

def calculate_sentiment_correlation(sentiment_data, other_data):
    sentiment_data['sentiment_correlation'] = sentiment_data['sentiment_intensity'].rolling(window=10).corr(other_data)
    return sentiment_data

def calculate_sentiment_clustering(sentiment_data):
    embeddings = sentence_model.encode(sentiment_data['preprocessed_comment'].tolist())
    clustering = AgglomerativeClustering(n_clusters=5)
    sentiment_data['sentiment_cluster'] = clustering.fit_predict(embeddings)
    return sentiment_data

def calculate_sentiment_propagation(sentiment_data):
    sentiment_data['sentiment_propagation'] = sentiment_data['sentiment_intensity'].diff().abs()
    return sentiment_data

def calculate_sentiment_echo_chambers(sentiment_data):
    sentiment_data['sentiment_echo_chamber'] = sentiment_data['sentiment_cluster'].rolling(window=10).apply(lambda x: len(set(x)) == 1)
    return sentiment_data

def calculate_sentiment_shifts(sentiment_data):
    sentiment_data['sentiment_shift'] = sentiment_data['sentiment_intensity'].diff()
    return sentiment_data

def calculate_sentiment_spikes(sentiment_data):
    sentiment_data['sentiment_spike'] = sentiment_data['sentiment_intensity'].diff().abs() > 0.5
    return sentiment_data

def calculate_sentiment_engagement(sentiment_data, engagement_data):
    sentiment_data['sentiment_engagement'] = engagement_data
    return sentiment_data

def calculate_sentiment_influence(sentiment_data, influence_data):
    sentiment_data['sentiment_influence'] = influence_data
    return sentiment_data
