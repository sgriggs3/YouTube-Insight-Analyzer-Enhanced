import youtube_api
import json

def test_get_comments_success():
    video_id = "glup6n0iHZk"  # User provided video ID
    comments = youtube_api.get_video_comments(video_id, comment_limit=20)
    assert comments is not None
    assert len(comments) > 0
    for comment in comments:
        assert "text" in comment
        assert "author" in comment
        assert "timestamp" in comment
    print("Successfully scraped comments with author and timestamp.")

def test_get_comments_invalid_video_id():
    video_id = "invalid_video_id"
    comments = youtube_api.get_video_comments(video_id, comment_limit=20)
    assert comments == []
    print("Successfully handled invalid video ID.")

def test_get_comments_invalid_api_key():
    # Temporarily modify config to use an invalid API key
    original_config = youtube_api.load_config()
    config = original_config.copy()
    config["youtube_api_key"] = "invalid_api_key"
    with open("config.json", "w") as f: # Use write_to_file tool later if this fails
        json.dump(config, f)

    comments = youtube_api.get_video_comments("glup6n0iHZk", comment_limit=20)
    assert comments == []
    print("Successfully handled invalid API key.")

    # Restore original config
    with open("config.json", "w") as f: # Use write_to_file tool later if this fails
        json.dump(original_config, f)


if __name__ == "__main__":
    test_get_comments_success()
    test_get_comments_invalid_video_id()
    test_get_comments_invalid_api_key()
    print("All tests completed.")