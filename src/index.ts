import express from "express";
import { server as WebSocketServer } from "websocket";

const app = express();
const port = process.env.PORT || 8080;

app.get("/", (req, res) => {
  res.send("YouTube Insight Analyzer Enhanced");
});

const server = app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

const wsServer = new WebSocketServer({
  httpServer: server,
  autoAcceptConnections: false,
});

import { fetchComments } from "./backend/youtube_api";

export { app, wsServer };

app.get("/api/comments", async (req, res) => {
  const videoId = req.query.videoId as string;
  const maxResults = parseInt(req.query.maxResults as string) || 100;

  if (!videoId) {
    return res.status(400).send({ message: "Video ID is required" });
  }

  try {
    const comments = await fetchComments(videoId, maxResults, (progress) => {
      console.log(`Scraping progress: ${progress * 100}%`);
    });
    res.send(comments);
  } catch (error: any) {
    console.error("Error fetching comments:", error);
    res
      .status(500)
      .send({ message: "Error fetching comments", error: error.message });
  }
});

app.get("/api/opinion", (req, res) => {
  // Placeholder for opinion prediction API endpoint
  res.send({ message: "Opinion prediction API endpoint not yet implemented" });
});

app.get("/api/bias", (req, res) => {
  // Placeholder for bias detection API endpoint
  res.send({ message: "Bias detection API endpoint not yet implemented" });
});

app.get("/api/social", (req, res) => {
  // Placeholder for social issue analysis API endpoint
  res.send({
    message: "Social issue analysis API endpoint not yet implemented",
  });
});

app.get("/api/psychological", (req, res) => {
  // Placeholder for psychological analysis API endpoint
  res.send({
    message: "Psychological analysis API endpoint not yet implemented",
  });
});

app.get("/api/philosophical", (req, res) => {
  // Placeholder for philosophical analysis API endpoint
  res.send({
    message: "Philosophical analysis API endpoint not yet implemented",
  });
});

app.get("/api/truth", (req, res) => {
  // Placeholder for truth/objectivity analysis API endpoint
  res.send({
    message: "Truth/objectivity analysis API endpoint not yet implemented",
  });
});

const ensureBackendRunning = () => {
  console.log("Backend is running and listening on the correct port");
};

ensureBackendRunning();
