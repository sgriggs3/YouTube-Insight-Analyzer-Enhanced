import { useCallback, useEffect, useState } from "react";
import { useEvent } from "react-use";
import ChatView from "./components/chat/ChatView.tsx";
import HistoryView from "./components/history/HistoryView.tsx";
import SettingsView from "./components/settings/SettingsView.tsx";
import WelcomeView from "./components/welcome/WelcomeView.tsx";
import { useExtensionState, ExtensionStateContextProvider } from "./context/ExtensionStateContext.tsx";
import { vscode } from "./utils/vscode.ts";
import McpView from "./components/mcp/McpView.tsx";
import AnalysisView from "./components/analysis/AnalysisView.tsx";

// Add state for video metadata
const App = () => {
  const [videoMetadata, setVideoMetadata] = useState(null);

const AppContent = () => {
  const { didHydrateState, showWelcome, shouldShowAnnouncement } = useExtensionState();
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showMcp, setShowMcp] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(true);
  const [showAnnouncement, setShowAnnouncement] = useState(false);

  const handleMessage = useCallback((e) => {
    const message = e.data;
    switch (message.type) {
      case "action":
        switch (message.action) {
          case "settingsButtonClicked":
            setShowSettings(true);
            setShowHistory(false);
            setShowMcp(false);
            break;
          case "historyButtonClicked":
            setShowSettings(false);
            setShowHistory(true);
            setShowMcp(false);
            break;
          case "mcpButtonClicked":
            setShowSettings(false);
            setShowHistory(false);
            setShowMcp(true);
            break;
          case "chatButtonClicked":
            setShowSettings(false);
            setShowHistory(false);
            setShowMcp(false);
            break;
        }
        break;
    }
  }, []);

  useEvent("message", handleMessage);

  useEffect(() => {
    if (shouldShowAnnouncement) {
      setShowAnnouncement(true);
      vscode.postMessage({ type: "didShowAnnouncement" });
    }
  }, [shouldShowAnnouncement]);

  // Add useEffect to fetch video metadata from backend API
  useEffect(() => {
    const fetchVideoMetadata = async () => {
      try {
        const videoId = extractVideoIdFromURL(window.location.href);
        if (!videoId) {
          throw new Error("Invalid video ID");
        }
        console.log("Fetching video metadata for video ID:", videoId);
        const response = await fetch(`/api/video-metadata/${videoId}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch video metadata: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        console.log("Video metadata fetched successfully:", data);
        setVideoMetadata(data);
      } catch (error) {
        console.error("Error fetching video metadata:", error);
      }
    };

    fetchVideoMetadata();
  }, []);

  if (!didHydrateState) {
    return null;
  }

  return (
    <>
      {showWelcome ? (
        <WelcomeView />
      ) : (
        <>
          {showSettings && <SettingsView onDone={() => setShowSettings(false)} />}
          {showHistory && <HistoryView onDone={() => setShowHistory(false)} />}
          {showMcp && <McpView onDone={() => setShowMcp(false)} />}
          {showAnalysis && <AnalysisView metadata={videoMetadata} onDone={() => setShowAnalysis(false)} />}
          {/* Do not conditionally load ChatView, it's expensive and there's state we don't want to lose (user input, disableInput, askResponse promise, etc.) */}
          <ChatView
            showHistoryView={() => {
              setShowSettings(false);
              setShowMcp(false);
              setShowAnalysis(false);
              setShowHistory(true);
            }}
            isHidden={showSettings || showHistory || showMcp || showAnalysis}
            showAnnouncement={showAnnouncement}
            hideAnnouncement={() => {
              setShowAnnouncement(false);
            }}
          />
        </>
      )}
    </>
  );
};

const extractVideoIdFromURL = (url) => {
  try {
    const urlObj = new URL(url);
    const params = new URLSearchParams(urlObj.search);
    return params.get("video_id");
  } catch (error) {
    console.error("Error extracting video ID:", error);
    return null;
  }
};

  return (
    <ExtensionStateContextProvider>
      <AppContent />
    </ExtensionStateContextProvider>
  );
};

export default App;
