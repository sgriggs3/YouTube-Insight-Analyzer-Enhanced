import unittest
from webui import app


class TestWebUI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_input_url_valid(self):
        response = self.app.post(
            "/input-url", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("URL processed successfully", response.json["message"])

    def test_input_url_invalid(self):
        response = self.app.post(
            "/input-url", json={"url": "https://www.youtube.com/watch?v=INVALID_ID"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid YouTube URL", response.json["error"])


if __name__ == "__main__":
    unittest.main()
