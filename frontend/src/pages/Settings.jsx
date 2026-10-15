import { useEffect, useState } from "react";
import toast from "react-hot-toast";
import { CheckCircle2, RefreshCw, Trash2 } from "lucide-react";
import { api, checkHealth, clearResults, getStatus } from "../api";

export default function Settings() {
  const [health, setHealth] = useState("checking");
  const [status, setStatus] = useState(null);

  async function refresh() {
    setHealth("checking");
    try {
      const healthData = await checkHealth();
      if (healthData?.status === "ok") {
        setHealth("online");
      } else {
        setHealth("offline");
      }

      try {
        const jobStatus = await getStatus();
        setStatus(jobStatus);
      } catch (statusError) {
        console.warn("Could not fetch pipeline job status:", statusError);
      }
    } catch (healthError) {
      console.error("Backend health check failed:", healthError);
      setHealth("offline");
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function clear() {
    try {
      await clearResults();
      toast.success("Results cleared");
      await refresh();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Could not clear results");
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <p className="text-sm font-semibold uppercase tracking-[0.22em] text-teal-500">Settings</p>
      <h1 className="mt-3 text-4xl font-semibold">System controls</h1>
      <div className="mt-8 grid gap-5 md:grid-cols-2">
        <section className="glass-card p-6">
          <CheckCircle2 className={health === "online" ? "text-teal-400" : "text-red-400"} />
          <h2 className="mt-4 text-xl font-semibold">Backend API</h2>
          <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">Status: {health}</p>
          <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">Current job: {status?.status || "--"}</p>
          <button className="secondary-btn mt-5" onClick={refresh}><RefreshCw size={18} /> Refresh</button>
        </section>
        <section className="glass-card p-6">
          <Trash2 className="text-red-400" />
          <h2 className="mt-4 text-xl font-semibold">Results</h2>
          <p className="mt-2 text-sm text-zinc-500 dark:text-zinc-400">Clear generated artifacts from the `results/` directory when no job is running.</p>
          <button className="danger-btn mt-5" onClick={clear}><Trash2 size={18} /> Clear Results</button>
        </section>
      </div>
      <section className="glass-card mt-5 p-6">
        <h2 className="text-xl font-semibold">Model Information</h2>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {["SpeechBrain SepFormer", "microsoft/wavlm-base-plus", "Spectral Clustering"].map((model) => (
            <div className="rounded-lg border border-black/10 bg-white/50 p-4 text-sm font-medium dark:border-white/10 dark:bg-white/[0.04]" key={model}>{model}</div>
          ))}
        </div>
      </section>
    </div>
  );
}
