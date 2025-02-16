from setuptools import setup, find_packages

setup(
    name="youtube-insight-analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask==2.0.1",
        "pandas==1.3.3",
        "plotly==5.3.1",
        "nltk==3.6.3",
        "vaderSentiment==3.3.2",
        "transformers==4.11.3",
        "torch==1.9.1",
        "scikit-learn==0.24.2",
        "spellchecker==0.4",
        "textblob==0.15.3",
        "wordcloud==1.8.1",
        "msgpack==1.0.2",
        "websockets==10.0",
        "google-generativeai==0.4.0",
        "flask-cors==3.0.10",
        "flask-compress==1.10.1",
        "google-api-python-client>=2.0.0",
        "python-dotenv>=0.19.0",
    ],
)
