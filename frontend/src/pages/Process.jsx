import { useCallback, useEffect, useMemo, useState } from "react";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import { motion } from "framer-motion";
import { AudioWaveform, Bot, Loader2, PlayCircle, Split, UploadCloud } from "lucide-react";
import { getLogs, getStatus, runJob, uploadAudio, uploadUrl } from "../api";
import WaveformPlayer from "../components/WaveformPlayer";

export default function Process() {
  const [upload, setUpload] = useState(null);
  const [status, setStatus] = useState({ status: "idle", stage: "Waiting for audio", progress: 0 });
  const [logs, setLogs] = useState([]);

  const refresh = useCallback(async () => {
    const [nextStatus, nextLogs] = await Promise.all([getStatus(), getLogs()]);
    setStatus(nextStatus);
    setLogs(nextLogs);
  }, []);

  useEffect(() => {
    refresh().catch(() => undefined);
    const interval = window.setInterval(() => refresh().catch(() => undefined), 1200);
    return () => window.clearInterval(interval);
  }, [refresh]);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;
    try {
      const response = await uploadAudio(file);
      setUpload({ ...response, localName: file.name, src: uploadUrl(response.storedName) });
      toast.success("Audio uploaded");
      await refresh();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Upload failed");
    }
  }, [refresh]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "audio/*": [".wav", ".mp3", ".m4a"] },
    multiple: false,
  });

  const isProcessing = status.status === "running";
  const canRun = Boolean(upload || status.uploaded_file) && !isProcessing;

  const actions = useMemo(() => [
    { key: "separation", label: "Run Speech Separation", icon: Split },
    { key: "diarization", label: "Run Speaker Diarization", icon: AudioWaveform },
    { key: "transcription", label: "Run Transcript Only", icon: PlayCircle },
    { key: "complete", label: "Run Complete Pipeline", icon: Bot },
  ], []);

  async function start(kind) {
    try {
      await runJob(kind);
      toast.success("Processing started");
      await refresh();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Could not start processing");
    }
  }

  return (
    <div className="mx-auto grid max-w-7xl gap-6 px-4 py-12 sm:px-6 lg:grid-cols-[1fr_.85fr]">
      <section className="space-y-6">
        <motion.div className="glass-card p-6 sm:p-8" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <div {...getRootProps()} className={`grid min-h-72 cursor-pointer place-items-center rounded-xl border border-dashed p-8 text-center transition ${isDragActive ? "border-teal-400 bg-teal-400/10" : "border-black/15 bg-white/45 dark:border-white/15 dark:bg-white/[0.03]"}`}>
            <input {...getInputProps()} />
            <div>
              <UploadCloud className="mx-auto text-teal-400" size={48} />
              <h1 className="mt-5 text-3xl font-semibold">Upload Audio</h1>
              <p className="mt-2 text-zinc-500 dark:text-zinc-400">Drag and drop, or click to upload wav, mp3, or m4a.</p>
              <p className="mt-4 text-sm font-medium text-zinc-700 dark:text-zinc-200">{upload?.localName || status.uploaded_file || "No file selected"}</p>
            </div>
          </div>
          {(upload?.src || status.uploaded_file) && upload?.src ? (
            <div className="mt-6">
              <WaveformPlayer src={upload.src} />
            </div>
          ) : null}
        </motion.div>
        <div className="grid gap-3 sm:grid-cols-3">
          {actions.map((action) => (
            <button key={action.key} className="secondary-btn min-h-14" disabled={!canRun} onClick={() => start(action.key)}>
              {isProcessing ? <Loader2 className="animate-spin" size={18} /> : <action.icon size={18} />}
              <span>{action.label}</span>
            </button>
          ))}
        </div>
      </section>
      <aside className="space-y-6">
        <section className="glass-card p-6">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">Current Stage</p>
              <h2 className="mt-1 text-xl font-semibold">{status.stage}</h2>
            </div>
            {isProcessing ? <Loader2 className="animate-spin text-teal-400" size={28} /> : <PlayCircle className="text-teal-400" size={28} />}
          </div>
          <div className="mt-6 h-3 overflow-hidden rounded-full bg-zinc-200 dark:bg-white/10">
            <div className="h-full rounded-full bg-gradient-to-r from-teal-400 to-sky-400 transition-all duration-500" style={{ width: `${status.progress || 0}%` }} />
          </div>
          <div className="mt-2 flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
            <span>{status.status}</span>
            <span>{status.progress || 0}%</span>
          </div>
          {status.error ? <p className="mt-4 rounded-lg bg-red-500/10 p-3 text-sm text-red-500 dark:text-red-300">{status.error}</p> : null}
        </section>
        <section className="glass-card p-6">
          <h2 className="text-xl font-semibold">Realtime Logs</h2>
          <div className="mt-4 h-80 overflow-auto rounded-lg bg-zinc-950 p-4 font-mono text-xs leading-6 text-zinc-200">
            {logs.length ? logs.map((line) => <p key={line}>{line}</p>) : <p className="text-zinc-500">Logs will appear here after upload.</p>}
          </div>
        </section>
      </aside>
    </div>
  );
}
