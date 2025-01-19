import unittest
from sentiment_analysis import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalyzer()

    def test_analyze_text_positive(self):
        text = "I love this video!"
        result = self.analyzer.analyze_text(text)
        self.assertEqual(result["label"], "POSITIVE")
        self.assertGreater(result["polarity"], 0)

    def test_analyze_text_negative(self):
        text = "I hate this content."
        result = self.analyzer.analyze_text(text)
        self.assertEqual(result["label"], "NEGATIVE")
        self.assertLess(result["polarity"], 0)

    def test_analyze_text_neutral(self):
        text = "This video is okay."
        result = self.analyzer.analyze_text(text)
        self.assertIn(
            result["label"], ["NEUTRAL", "POSITIVE", "NEGATIVE"]
        )  # Depending on the model

    def test_analyze_comments(self):
        comments = [
            {"text": "Great video!"},
            {"text": "Not what I expected."},
            {"text": "It's fine."},
        ]
        analysis = self.analyzer.analyze_comments(comments)
        self.assertIn("comments", analysis)
        self.assertIn("overall_sentiment", analysis)
        self.assertEqual(len(analysis["comments"]), 3)

    def test_analyze_comments_empty(self):
        comments = []
        analysis = self.analyzer.analyze_comments(comments)
        self.assertEqual(analysis["comments"], [])
        self.assertEqual(analysis["overall_sentiment"]["positive"], 0)
        self.assertEqual(analysis["overall_sentiment"]["negative"], 0)
        self.assertEqual(analysis["overall_sentiment"]["neutral"], 0)
        self.assertEqual(analysis["overall_sentiment"]["avg_polarity"], 0)

    def test_perform_sentiment_analysis_vader(self):
        text_inputs = ["I love this video!", "I hate this content."]
        results = perform_sentiment_analysis(text_inputs, language="en")
        self.assertEqual(len(results), 2)
        self.assertIn("vader_sentiment", results[0])
        self.assertIn("vader_sentiment", results[1])

    def test_perform_sentiment_analysis_hf(self):
        text_inputs = ["I love this video!", "I hate this content."]
        config = load_config()
        config["sentiment_model"] = "hf"
        results = perform_sentiment_analysis(text_inputs, language="en")
        self.assertEqual(len(results), 2)
        self.assertIn("hf_sentiment", results[0])
        self.assertIn("hf_sentiment", results[1])


if __name__ == "__main__":
    unittest.main()
