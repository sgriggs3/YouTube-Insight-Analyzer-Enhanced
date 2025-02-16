import React, { useEffect, useState, useCallback } from "react";
import { vscode } from "../../utils/vscode";
import { VSCodeProgressRing } from "@vscode/webview-ui-toolkit/react";

interface VideoMetadata {
  title: string;
  description: string;
  views: number;
  likes: number;
  publishedAt: string;
}

interface Comment {
  text: string;
  author: string;
  timestamp: string;
  sentiment: string;
}

interface AnalysisViewProps {
  videoId: string | null;
  onDone: () => void;
}

const AnalysisView: React.FC<AnalysisViewProps> = ({ videoId, onDone }) => {
  const [metadata, setMetadata] = useState<VideoMetadata | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [analysisState, setAnalysisState] = useState<
    "idle" | "loading" | "error" | "success"
  >("idle");
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [comments, setComments] = useState<Comment[]>([]);

  const handleRetry = useCallback(() => {
    setRetryCount((count) => count + 1);
    setError(null);
  }, []);

  const fetchWithTimeout = async (url: string, timeout = 5000) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, { signal: controller.signal });
      clearTimeout(timeoutId);
      return response;
    } catch (err) {
      clearTimeout(timeoutId);
      throw err;
    }
  };

  const handleAnalyzeClick = async () => {
    setProgress(0);
    setProgressMessage("");
    setAnalysisState("loading");
    try {
      // Placeholder for analysis logic
      // Replace with actual analysis calls in later iterations
      await new Promise((resolve) => setTimeout(resolve, 2000));
      setAnalysisState("success");
    } catch (error) {
      console.error("Error during analysis:", error);
      setAnalysisState("error");
      setError("Analysis failed. Please try again.");
    }
  };

  const handleScrape = async () => {
    if (!videoId) {
      return;
    }
    setProgress(0);
    setProgressMessage("Scraping comments...");
    try {
      const response = await fetch(
        `/api/comments?videoId=${videoId}&maxResults=500`
      );
      if (!response.ok) {
        throw new Error(
          `Failed to scrape comments: ${response.status} ${response.statusText}`
        );
      }
      const comments = await response.json();
      setComments(comments);
    } catch (error) {
      console.error("Error scraping comments:", error);
    }
  };

  useEffect(() => {
    const updateProgress = (message: {
      type: string;
      progress: number;
      message: string;
    }) => {
      if (message.type === "progress") {
        setProgress(message.progress * 100);
        setProgressMessage(message.message);
      }
    };
    vscode.postMessage({ type: "registerProgressListener" });
    window.addEventListener("message", (event: MessageEvent) => {
      updateProgress(
        event.data as { type: string; progress: number; message: string }
      );
    });
    return () => {
      vscode.postMessage({ type: "unregisterProgressListener" });
      window.removeEventListener("message", updateProgress);
    };
  }, []);

  useEffect(() => {
    if (!videoId) return;

    const fetchMetadata = async () => {
      setLoading(true);
      try {
        const response = await fetch(`/api/video-metadata/${videoId}`);
        if (!response.ok) {
          throw new Error("Failed to fetch video metadata");
        }
        const data = await response.json();
        setMetadata(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An error occurred");
      } finally {
        setLoading(false);
      }
    };

    fetchMetadata();
  }, [videoId]);

  if (analysisState === "loading") {
    return (
      <div className="loading-container">
        <VSCodeProgressRing />
        <span>Analyzing video content...</span>
      </div>
    );
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  if (!videoId || !comments) {
    return <div>No video ID or comments available</div>;
  }

  if (comments.length === 0 && metadata === null) {
    return <div>No comments available. Please scrape some comments.</div>;
  }

  return (
    <div className="analysis-container">
      <h2>{metadata?.title}</h2>
      <div className="metadata-grid">
        <div className="metadata-item">
          <label>Views:</label>
          <span>{metadata?.views.toLocaleString()}</span>
        </div>
        <div className="metadata-item">
          <label>Likes:</label>
          <span>{metadata?.likes.toLocaleString()}</span>
        </div>
        <div className="metadata-item">
          <label>Published:</label>
          <span>
            {metadata?.publishedAt
              ? new Date(metadata.publishedAt).toLocaleDateString()
              : ""}
          </span>
        </div>
      </div>
      <p className="description">{metadata?.description}</p>
      <h2>Analysis</h2>
      <ul>
        {comments.map((comment) => (
          <li key={comment.timestamp}>
            <p>{comment.text}</p>
            <p>Sentiment: {comment.sentiment}</p>
            <p>Author: {comment.author}</p>
            <p>Timestamp: {new Date(comment.timestamp).toLocaleString()}</p>
          </li>
        ))}
      </ul>
      <div className="scraping-options">
        <label htmlFor="comment-limit">Comment Limit:</label>
        <input type="number" id="comment-limit" defaultValue={500} />
        <button onClick={handleScrape} disabled={loading}>
          Scrape Comments
        </button>
      </div>
      <div
        className="progress-container"
        style={{ display: progress > 0 ? "block" : "none" }}
      >
        <div className="progress-bar" style={{ width: `${progress}%` }}>
          {progress}% - {progressMessage}
        </div>
      </div>
    </div>
  );
};

export default AnalysisView;
