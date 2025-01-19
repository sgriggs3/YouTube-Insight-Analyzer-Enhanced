import express from 'express';
import { server as WebSocketServer } from 'websocket';

const app = express();
const port = process.env.PORT || 8080;

app.get('/', (req, res) => {
  res.send('YouTube Insight Analyzer Enhanced');
});

const server = app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

const wsServer = new WebSocketServer({
  httpServer: server,
  autoAcceptConnections: false
});

export { app, wsServer };
