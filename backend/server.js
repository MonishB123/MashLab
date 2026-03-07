import { createServer } from "node:http";

const PORT = Number(process.env.PORT || 8787);
const ALLOWED_ORIGIN = process.env.FRONTEND_ORIGIN || "http://localhost:5173";

const sendJson = (res, status, payload) => {
  const origin = ALLOWED_ORIGIN === "*" ? "*" : ALLOWED_ORIGIN;
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  });
  res.end(JSON.stringify(payload));
};

const readJsonBody = (req) =>
  new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 1_000_000) {
        reject(new Error("Body too large"));
      }
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        reject(new Error("Invalid JSON body"));
      }
    });
    req.on("error", reject);
  });

const analyzeSongs = ({ song1Name, song2Name }) => {
  const score = Math.floor(Math.random() * 101);
  const preview_source = Math.random() < 0.5 ? "song1" : "song2";
  const preview_song_name = (preview_source === "song1" ? song1Name : song2Name)?.replace(
    /\.[^/.]+$/,
    ""
  );

  return {
    score,
    preview_source,
    preview_song_name: preview_song_name || "Mashup Preview",
  };
};

const server = createServer(async (req, res) => {
  if (!req.url || !req.method) {
    sendJson(res, 400, { error: "Invalid request" });
    return;
  }

  if (req.method === "OPTIONS") {
    sendJson(res, 200, { ok: true });
    return;
  }

  if (req.method === "GET" && req.url === "/api/health") {
    sendJson(res, 200, { ok: true, service: "mashlab-backend" });
    return;
  }

  if (req.method === "POST" && req.url === "/api/analyze") {
    try {
      const body = await readJsonBody(req);
      if (!body.song1Name || !body.song2Name) {
        sendJson(res, 400, { error: "song1Name and song2Name are required" });
        return;
      }

      await new Promise((resolve) => setTimeout(resolve, 1200));
      sendJson(res, 200, analyzeSongs(body));
      return;
    } catch (error) {
      sendJson(res, 400, { error: error.message || "Unable to process request" });
      return;
    }
  }

  sendJson(res, 404, { error: "Not found" });
});

server.listen(PORT, () => {
  console.log(`MashLab backend listening on http://localhost:${PORT}`);
});
