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
    transcription: "/run-transcription",
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

// --- Diarization evaluation (DER / JER) ---

export const evaluationDownloadUrl = (path) =>
  `${apiBase}/evaluation/download/${path.split("/").map(encodeURIComponent).join("/")}`;

export async function getEvaluationFiles() {
  const response = await api.get("/evaluation/files");
  return response.data;
}

export async function runEvaluation() {
  const response = await api.post("/run-evaluation");
  return response.data;
}

export async function getEvaluationStatus() {
  const response = await api.get("/evaluation/status");
  return response.data;
}

export async function getEvaluationLogs() {
  const response = await api.get("/evaluation/logs");
  return response.data.logs;
}

export async function getEvaluationResults() {
  const response = await api.get("/evaluation/results");
  return response.data;
}

export async function getEvaluationResult(name) {
  const response = await api.get(`/evaluation/results/${encodeURIComponent(name)}`);
  return response.data;
}

export async function clearEvaluationResults() {
  const response = await api.delete("/evaluation/results");
  return response.data;
}
