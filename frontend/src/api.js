import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "/api",
  timeout: 120000,
});

const apiBase = api.defaults.baseURL.replace(/\/$/, "");

export const downloadUrl = (path) => `${apiBase}/download/${encodeURI(path)}`;
export const uploadUrl = (storedName) => `${apiBase}/download-upload/${storedName.split("/").map(encodeURIComponent).join("/")}`;

export async function uploadAudio(file) {
  const data = new FormData();
  data.append("file", file);
  const response = await api.post("/upload", data, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

export async function runJob(kind) {
  const endpoint = {
    separation: "/run-separation",
    diarization: "/run-diarization",
    complete: "/run-complete",
  }[kind];
  const response = await api.post(endpoint);
  return response.data;
}

export async function getStatus() {
  const response = await api.get("/status");
  return response.data;
}

export async function getLogs() {
  const response = await api.get("/logs");
  return response.data.logs;
}

export async function getResults() {
  const response = await api.get("/results");
  return response.data;
}

export async function clearResults() {
  const response = await api.delete("/results");
  return response.data;
}
