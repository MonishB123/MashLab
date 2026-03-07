const importMeta = /** @type {any} */ (import.meta);
const viteEnv = /** @type {{ VITE_API_BASE_URL?: string } | undefined} */ (importMeta.env);

const API_BASE_URL = (viteEnv?.VITE_API_BASE_URL || "http://localhost:8787").replace(/\/$/, "");

export async function analyzeSongs(song1, song2) {
  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      song1Name: song1?.name,
      song2Name: song2?.name,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || `Backend request failed with ${response.status}`);
  }

  return response.json();
}
