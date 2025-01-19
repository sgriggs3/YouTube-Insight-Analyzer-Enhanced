import unittest
from youtube_api import YouTubeAPI


class TestYouTubeAPI(unittest.TestCase):
    def setUp(self):
        self.api = YouTubeAPI(api_key="TEST_API_KEY")

    def test_get_video_metadata_valid_id(self):
        video_id = "dQw4w9WgXcQ"  # Example video ID
        metadata = self.api.get_video_metadata(video_id)
        self.assertIn("title", metadata)
        self.assertIn("description", metadata)

    def test_get_video_metadata_invalid_id(self):
        video_id = "INVALID_ID"
        with self.assertRaises(HttpError):
            self.api.get_video_metadata(video_id)

    def test_get_video_comments_valid_id(self):
        video_id = "dQw4w9WgXcQ"
        comments = self.api.get_video_comments(video_id, max_results=10)
        self.assertIsInstance(comments, list)
        self.assertLessEqual(len(comments), 10)

    def test_get_video_transcript_valid_id(self):
        video_id = "dQw4w9WgXcQ"
        transcript = self.api.get_video_transcript(video_id)
        # Transcript may be None if not available
        self.assertTrue(isinstance(transcript, str) or transcript is None)


if __name__ == "__main__":
    unittest.main()
