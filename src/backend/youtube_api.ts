import { SentimentIntensityAnalyzer } from "vaderSentiment";
import axios, { AxiosError } from "axios";

interface Comment {
  text: string;
  author: string;
  timestamp: string;
}

interface YouTubeAPIResponse {
  items: {
    snippet: {
      topLevelComment: {
        snippet: {
          textDisplay: string;
          authorDisplayName: string;
          publishedAt: string;
        };
      };
    };
  }[];
  nextPageToken?: string;
}

const API_KEYS = [
  process.env.YOUTUBE_API_KEY_1,
  process.env.YOUTUBE_API_KEY_2,
  process.env.YOUTUBE_API_KEY_3,
];

let currentApiKeyIndex = 0;

const getApiKey = () => {
  return API_KEYS[currentApiKeyIndex];
};

const rotateApiKey = () => {
  currentApiKeyIndex = (currentApiKeyIndex + 1) % API_KEYS.length;
  console.log(`Rotating API key to index ${currentApiKeyIndex}`);
};

const logApiInteraction = (
  url: string,
  params: any,
  response: any,
  error?: any
) => {
  const logEntry = {
    timestamp: new Date().toISOString(),
    url,
    params,
    response,
    error: error ? error.message : null,
  };
  console.log(JSON.stringify(logEntry, null, 2));
};

const fetchData = async (url: string, params: any) => {
  try {
    const apiKey = getApiKey();
    const response = await axios.get(url, {
      params: { ...params, key: apiKey },
    });
    logApiInteraction(url, params, response.data);
    return response.data;
  } catch (error: any) {
    logApiInteraction(url, params, null, error);
    if (error instanceof AxiosError && error.response?.status === 403) {
      rotateApiKey();
      throw new Error("API rate limit exceeded, rotating API key and retrying");
    }
    throw error;
  }
};

const fetchComments = async (
  videoId: string,
  maxResults: number = 100,
  progressCallback?: (progress: number) => void
): Promise<Comment[]> => {
  let allComments: Comment[] = [];
  let nextPageToken: string | undefined;
  let totalResults = 0;
  const maxPages = Math.ceil(maxResults / 100);
  let currentPage = 0;

  while (totalResults < maxResults && currentPage < maxPages) {
    try {
      currentPage++;
      const url = "https://www.googleapis.com/youtube/v3/commentThreads";
      const params = {
        part: "snippet",
        videoId: videoId,
        maxResults: Math.min(100, maxResults - totalResults),
        pageToken: nextPageToken,
      };

      const response: YouTubeAPIResponse = await fetchData(url, params);

      if (response.items) {
        const comments: Comment[] = response.items.map((item) => {
          return {
            text: item.snippet.topLevelComment.snippet.textDisplay,
            author: item.snippet.topLevelComment.snippet.authorDisplayName,
            timestamp: item.snippet.topLevelComment.snippet.publishedAt,
          };
        });
        allComments.push(...comments);
        totalResults += comments.length;
      }

      nextPageToken = response.nextPageToken;

      if (progressCallback) {
        progressCallback(Math.min(1, totalResults / maxResults));
      }

      if (!nextPageToken) {
        break;
      }
    } catch (error: any) {
      console.error("Error fetching comments:", error);
      if (error.message.includes("API rate limit exceeded")) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        continue;
      }
      throw error;
    }
  }
  return allComments;
};

import fs from "fs/promises";
import path from "path";

const saveComments = async (comments: Comment[]) => {
  const filePath = path.join(__dirname, "../data/comments.json");
  try {
    const existingData = await fs.readFile(filePath, "utf-8");
    const existingComments = JSON.parse(existingData);
    const updatedComments = [...existingComments, ...comments];
    await fs.writeFile(filePath, JSON.stringify(updatedComments, null, 2));
  } catch (error) {
    console.error("Error saving comments:", error);
    await fs.writeFile(filePath, JSON.stringify(comments, null, 2));
  }
};

import { getVideoMetadata } from "./video_metadata";

const analyzeCommentSentiment = (comment: string) => {
  // Basic sentiment analysis using VADER
  // Replace with more advanced analysis in later iterations
  const sentiment = SentimentIntensityAnalyzer().polarity_scores(comment);
  let overallSentiment;
  if (sentiment.compound >= 0.05) {
    overallSentiment = "positive";
  } else if (sentiment.compound <= -0.05) {
    overallSentiment = "negative";
  } else {
    overallSentiment = "neutral";
  }
  return overallSentiment;
};

export { fetchComments, analyzeCommentSentiment };
