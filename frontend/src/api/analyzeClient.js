const importMeta = /** @type {any} */ (import.meta);
const viteEnv = /** @type {{ VITE_API_BASE_URL?: string } | undefined} */ (importMeta.env);
const API_BASE_URL = (viteEnv?.VITE_API_BASE_URL || "http://localhost:8787").replace(/\/$/, "");

const toAbsoluteUrl = (value) => {
  if (!value) return "";
  if (value.startsWith("http://") || value.startsWith("https://")) return value;
  return `${API_BASE_URL}${value.startsWith("/") ? "" : "/"}${value}`;
};

const stripExt = (name) => name?.replace(/\.[^/.]+$/, "") || "";

async function parseJson(response) {
  return response.json().catch(() => ({}));
}

async function parseErrorMessage(response, fallback) {
  const text = await response.text().catch(() => "");
  if (!text) return fallback;

  try {
    const json = JSON.parse(text);
    return json.detail || json.error || fallback;
  } catch {
    return text;
  }
}

async function request(url, options, stage) {
  try {
    return await fetch(url, options);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(
      `${stage} request could not reach backend (${detail}). Check ${API_BASE_URL} and backend logs.`
    );
  }
}

async function uploadTracks(song1, song2) {
  const formData = new FormData();
  formData.append("track_a", song1);
  formData.append("track_b", song2);

  const response = await request(`${API_BASE_URL}/api/upload`, {
    method: "POST",
    body: formData,
  }, "Upload");

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response, "Upload failed"));
  }

  return parseJson(response);
}

async function analyzeSession(sessionId) {
  const response = await request(`${API_BASE_URL}/api/analyze/${sessionId}`, {
    method: "POST",
  }, "Analyze");

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response, "Analysis failed"));
  }

  return parseJson(response);
}

async function createPreview(sessionId) {
  const response = await request(`${API_BASE_URL}/api/preview/${sessionId}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      clip_duration: 45,
      mashup_mode: "auto",
      use_stem_separation: true,
    }),
  }, "Preview");

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response, "Preview render failed"));
  }

  return parseJson(response);
}

export async function analyzeSongs(song1, song2) {
  if (!song1 || !song2) {
    throw new Error("Both tracks are required");
  }

  const uploadResult = await uploadTracks(song1, song2);
  const sessionId = uploadResult.session_id;
  if (!sessionId) {
    throw new Error("Backend did not return a session_id");
  }

  const analysisResult = await analyzeSession(sessionId);
  const previewResult = await createPreview(sessionId);

  return {
    score: Number(analysisResult?.compatibility?.score ?? 0),
    preview_url: toAbsoluteUrl(previewResult.audio_url),
    preview_song_name: `${stripExt(song1.name)} x ${stripExt(song2.name)}`,
    session_id: sessionId,
    compatibility: analysisResult?.compatibility,
    preview: previewResult,
  };
}
