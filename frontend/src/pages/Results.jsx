import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import toast from "react-hot-toast";
import Skeleton from "react-loading-skeleton";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Download, FileArchive, Gauge, Image as ImageIcon, RefreshCw } from "lucide-react";
import { downloadUrl, evaluationDownloadUrl, getEvaluationResults, getResults, uploadUrl } from "../api";
import EmptyState from "../components/EmptyState";
import MetricCard from "../components/MetricCard";
import WaveformPlayer from "../components/WaveformPlayer";

export default function Results() {
  const [results, setResults] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      setResults(await getResults());
    } catch {
      setResults(null);
    } finally {
      setLoading(false);
    }
    try {
      setEvaluation(await getEvaluationResults());
    } catch {
      setEvaluation(null);
    }
  }

  useEffect(() => {
    load();
  }, []);

  const chartData = useMemo(() => {
    if (!results?.separated_audio?.length) return [];
    return results.separated_audio.map((file, index) => ({
      speaker: `Speaker ${index + 1}`,
      size: Math.round(file.size / 1024),
    }));
  }, [results]);

  if (loading) {
    return <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6"><Skeleton height={420} baseColor="#18181b" highlightColor="#27272a" /></div>;
  }

  if (!results) {
    return <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6"><EmptyState title="No results yet" body="Upload audio and run a pipeline from the Process page to populate this dashboard." /></div>;
  }

  const stats = results.statistics || {};
  const timelineImage = results.figures.find((file) => file.name === "timeline.png");
  const activityImage = results.figures.find((file) => file.name === "speaker_activity.png");
  const histogramImage = results.figures.find((file) => file.name === "duration_histogram.png");

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-12 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.22em] text-teal-500">Results</p>
          <h1 className="mt-3 text-4xl font-semibold">Analysis dashboard</h1>
        </div>
        <div className="flex gap-2">
          <button className="secondary-btn" onClick={() => load()}><RefreshCw size={18} /> Refresh</button>
          <a
            className="primary-btn"
            href={downloadUrl(results.files.find((file) => file.name === "results.zip")?.path || "results.zip")}
            onClick={() => toast.success("Preparing download")}
          ><FileArchive size={18} /> Download All</a>
        </div>
      </div>

      <section className="glass-card p-6">
        <h2 className="text-xl font-semibold">Original Audio</h2>
        {results.original_audio ? <div className="mt-4"><WaveformPlayer src={uploadUrl(results.original_audio.path)} /></div> : <p className="mt-3 text-sm text-zinc-500">Original upload is not available.</p>}
      </section>

      <section>
        <h2 className="mb-4 text-xl font-semibold">Speech Separation</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          {results.separated_audio.length ? results.separated_audio.map((file, index) => (
            <article className="glass-card p-5" key={file.path}>
              <div className="mb-4 flex items-center justify-between">
                <h3 className="font-semibold">Speaker {index + 1}</h3>
                <a className="icon-btn" href={downloadUrl(file.path)} aria-label={`Download ${file.name}`}><Download size={18} /></a>
              </div>
              <WaveformPlayer compact src={downloadUrl(file.path)} />
            </article>
          )) : <EmptyState title="No separated tracks" body="Run speech separation or the complete pipeline to generate speaker audio files." />}
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <MetricCard label="Number of Speakers" value={stats.numberOfSpeakers} />
        <MetricCard label="Speech Segments" value={stats.speechSegments} />
        <MetricCard label="Embedding Dimension" value={stats.embeddingDimension} />
        <MetricCard label="Processing Time" value={stats.processingTime ? `${stats.processingTime}s` : "--"} />
        <MetricCard label="Models Used" value={Array.isArray(stats.modelsUsed) ? stats.modelsUsed.length : "--"} detail={Array.isArray(stats.modelsUsed) ? stats.modelsUsed.join(", ") : ""} />
        <MetricCard label="Output Files Generated" value={stats.outputFilesGenerated} />
      </section>

      <section className="grid gap-5 lg:grid-cols-2">
        <FigureCard title="Speaker Timeline" file={timelineImage} />
        <FigureCard title="Speaker Activity Chart" file={activityImage} />
        <FigureCard title="Duration Histogram" file={histogramImage} />
        <div className="glass-card p-5">
          <h2 className="text-xl font-semibold">Speaker Activity</h2>
          <div className="mt-4 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,.25)" />
                <XAxis dataKey="speaker" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip />
                <Bar dataKey="size" fill="#2dd4bf" radius={[6, 6, 0, 0]} name="Audio KB" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="glass-card p-6">
        <h2 className="text-xl font-semibold">Transcript</h2>
        {results.transcript?.segments?.length ? (
          <div className="space-y-3">
            {results.transcript.segments.map((segment, index) => (
              <div key={`${segment.speaker}-${segment.start}-${index}`} className="rounded-2xl border border-zinc-200 bg-white/70 p-4 text-sm shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
                <p className="font-semibold text-zinc-800 dark:text-zinc-100">{segment.speaker}</p>
                <p className="text-xs uppercase tracking-[0.22em] text-zinc-500 dark:text-zinc-400">{segment.start.toFixed(2)}s - {segment.end.toFixed(2)}s</p>
                <p className="mt-2 text-zinc-700 dark:text-zinc-200">{segment.text || "[No recognized text]"}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-zinc-500">Transcript has not been generated yet. Run speaker diarization and transcription from the backend.</p>
        )}
      </section>

      <section className="glass-card p-6">
        <h2 className="text-xl font-semibold">Generated Files</h2>
        <div className="mt-5 grid gap-3 md:grid-cols-2">
          {results.files.map((file) => (
            <a key={file.path} className="flex items-center justify-between rounded-lg border border-black/10 bg-white/50 p-3 transition hover:bg-white dark:border-white/10 dark:bg-white/[0.04] dark:hover:bg-white/[0.08]" href={downloadUrl(file.path)}>
              <span className="min-w-0 truncate text-sm font-medium">{file.path}</span>
              <Download className="shrink-0 text-teal-400" size={18} />
            </a>
          ))}
        </div>
      </section>

      <EvaluationSection payload={evaluation} />
    </div>
  );
}

function EvaluationSection({ payload }) {
  const records = payload?.results || [];
  const summary = payload?.summary || null;
  const downloads = payload?.downloads || {};

  return (
    <section className="glass-card p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-xl font-semibold">
          <Gauge className="text-teal-400" size={20} /> Evaluation
        </h2>
        <Link className="secondary-btn" to="/evaluation">
          Open Evaluation page
        </Link>
      </div>
      {records.length ? (
        <>
          {summary ? (
            <p className="mt-4 text-sm text-zinc-500 dark:text-zinc-400">
              Overall DER {formatPercent(summary.overall_der_percent)}
              {summary.jer_available ? ` · Overall JER ${formatPercent(summary.overall_jer_percent)}` : " · JER unavailable"}
              {" "}across {summary.file_count} file(s), duration-weighted. Scored with {summary.metrics_backend}
              {summary.metrics_backend_version ? ` ${summary.metrics_backend_version}` : ""} at collar{" "}
              {summary.protocol?.collar}s.
            </p>
          ) : null}
          <div className="mt-5 space-y-4">
            {records.map((record) => (
              <article
                key={record.name}
                className="rounded-lg border border-black/10 bg-white/50 p-5 dark:border-white/10 dark:bg-white/[0.04]"
              >
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <h3 className="font-semibold">{record.file}</h3>
                  <div className="flex gap-2">
                    {downloads[record.name]?.evaluation ? (
                      <a className="secondary-btn" href={evaluationDownloadUrl(downloads[record.name].evaluation)}>
                        <Download size={16} /> evaluation.json
                      </a>
                    ) : null}
                    {downloads[record.name]?.hypothesis ? (
                      <a className="secondary-btn" href={evaluationDownloadUrl(downloads[record.name].hypothesis)}>
                        <Download size={16} /> hypothesis.rttm
                      </a>
                    ) : null}
                  </div>
                </div>
                <dl className="mt-4 grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
                  <EvaluationField label="DER" value={formatPercent(record.der_percent)} highlight />
                  <EvaluationField
                    label="JER"
                    value={record.jer_available ? formatPercent(record.jer_percent) : "Unavailable"}
                    highlight={record.jer_available}
                  />
                  <EvaluationField label="Reference Speakers" value={record.reference_speakers ?? "--"} />
                  <EvaluationField label="Detected Speakers" value={record.detected_speakers ?? "--"} />
                  <EvaluationField label="False Alarm" value={formatSeconds(record.false_alarm)} />
                  <EvaluationField label="Missed Speech" value={formatSeconds(record.missed_speech)} />
                  <EvaluationField label="Speaker Confusion" value={formatSeconds(record.speaker_confusion)} />
                  <EvaluationField label="Duration" value={formatSeconds(record.duration)} />
                </dl>
              </article>
            ))}
          </div>
        </>
      ) : (
        <p className="mt-4 text-sm text-zinc-500">
          No diarization evaluation has been run yet. Place reference audio and RTTM files in the{" "}
          <code className="rounded bg-black/5 px-1 dark:bg-white/10">evaluation/</code> folder, then run the evaluation
          from the Evaluation page.
        </p>
      )}
    </section>
  );
}

function EvaluationField({ label, value, highlight }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-[0.16em] text-zinc-500 dark:text-zinc-400">{label}</dt>
      <dd className={`mt-1 font-semibold ${highlight ? "text-teal-600 dark:text-teal-300" : ""}`}>{value}</dd>
    </div>
  );
}

function formatPercent(value) {
  return value === null || value === undefined ? "--" : `${Number(value).toFixed(2)}%`;
}

function formatSeconds(value) {
  return value === null || value === undefined ? "--" : `${Number(value).toFixed(2)}s`;
}

function FigureCard({ title, file }) {
  return (
    <article className="glass-card p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">{title}</h2>
        {file ? <a className="icon-btn" href={downloadUrl(file.path)} aria-label={`Download ${file.name}`}><Download size={18} /></a> : <ImageIcon className="text-zinc-400" />}
      </div>
      {file ? <img className="mt-4 aspect-video w-full rounded-lg object-contain" src={downloadUrl(file.path)} alt={title} /> : <div className="mt-4 grid aspect-video place-items-center rounded-lg bg-white/5 text-sm text-zinc-500">Not generated yet</div>}
    </article>
  );
}
