import { useCallback, useEffect, useMemo, useState } from "react";
import toast from "react-hot-toast";
import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Download,
  FileAudio,
  Gauge,
  Loader2,
  PlayCircle,
  RefreshCw,
  Trash2,
} from "lucide-react";
import {
  clearEvaluationResults,
  evaluationDownloadUrl,
  getEvaluationFiles,
  getEvaluationLogs,
  getEvaluationResults,
  getEvaluationStatus,
  runEvaluation,
} from "../api";
import EmptyState from "../components/EmptyState";
import MetricCard from "../components/MetricCard";

const IDLE_STATUS = { status: "idle", stage: "Evaluation has not been run", progress: 0 };
const SORT_COLUMNS = {
  file: (row) => row.file || "",
  duration: (row) => row.duration ?? -1,
  reference_speakers: (row) => row.reference_speakers ?? -1,
  detected_speakers: (row) => row.detected_speakers ?? -1,
  der: (row) => row.der ?? -1,
  jer: (row) => (row.jer === null || row.jer === undefined ? -1 : row.jer),
};

export default function Evaluation() {
  const [dataset, setDataset] = useState(null);
  const [datasetError, setDatasetError] = useState(null);
  const [status, setStatus] = useState(IDLE_STATUS);
  const [logs, setLogs] = useState([]);
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [sort, setSort] = useState({ column: "der", direction: "asc" });

  const loadDataset = useCallback(async () => {
    try {
      setDataset(await getEvaluationFiles());
      setDatasetError(null);
    } catch (error) {
      setDataset(null);
      setDatasetError(error.response?.data?.detail || "Could not read the evaluation directory.");
    }
  }, []);

  const loadResults = useCallback(async () => {
    try {
      setPayload(await getEvaluationResults());
    } catch {
      setPayload(null);
    }
  }, []);

  const pollProgress = useCallback(async () => {
    const [nextStatus, nextLogs] = await Promise.all([getEvaluationStatus(), getEvaluationLogs()]);
    setStatus(nextStatus);
    setLogs(nextLogs);
    return nextStatus;
  }, []);

  const refreshAll = useCallback(async () => {
    await Promise.all([loadDataset(), loadResults(), pollProgress().catch(() => undefined)]);
  }, [loadDataset, loadResults, pollProgress]);

  useEffect(() => {
    refreshAll().finally(() => setLoading(false));
  }, [refreshAll]);

  const isRunning = status.status === "running";

  useEffect(() => {
    if (!isRunning) return undefined;
    const interval = window.setInterval(() => {
      pollProgress()
        .then((next) => {
          if (next.status !== "running") loadResults();
        })
        .catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(interval);
  }, [isRunning, pollProgress, loadResults]);

  const summary = payload?.summary || null;
  const records = payload?.results || [];
  const downloads = payload?.downloads || {};

  const rows = useMemo(() => {
    const accessor = SORT_COLUMNS[sort.column] || SORT_COLUMNS.der;
    const sorted = [...records].sort((left, right) => {
      const a = accessor(left);
      const b = accessor(right);
      if (typeof a === "string" || typeof b === "string") return String(a).localeCompare(String(b));
      return a - b;
    });
    return sort.direction === "asc" ? sorted : sorted.reverse();
  }, [records, sort]);

  const breakdownData = useMemo(
    () =>
      records.map((record) => ({
        file: record.name || record.file,
        "False Alarm": record.false_alarm ?? 0,
        "Missed Speech": record.missed_speech ?? 0,
        "Speaker Confusion": record.speaker_confusion ?? 0,
      })),
    [records],
  );

  const speakerData = useMemo(
    () =>
      records.map((record) => ({
        file: record.name || record.file,
        Reference: record.reference_speakers ?? 0,
        Detected: record.detected_speakers ?? 0,
      })),
    [records],
  );

  const derData = useMemo(
    () =>
      records.map((record) => ({
        file: record.name || record.file,
        der: record.der_percent ?? 0,
        jer: record.jer_percent,
      })),
    [records],
  );

  async function start() {
    try {
      const next = await runEvaluation();
      setStatus(next);
      setLogs([]);
      toast.success("Evaluation started");
      await pollProgress();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Could not start evaluation");
    }
  }

  async function clearAll() {
    try {
      const response = await clearEvaluationResults();
      setPayload(null);
      setStatus(IDLE_STATUS);
      setLogs([]);
      toast.success(`Cleared ${response.removed?.length ?? 0} generated file(s)`);
      await loadDataset();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Could not clear evaluation results");
    }
  }

  function toggleSort(column) {
    setSort((previous) =>
      previous.column === column
        ? { column, direction: previous.direction === "asc" ? "desc" : "asc" }
        : { column, direction: "asc" },
    );
  }

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-12 sm:px-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.22em] text-teal-500">Evaluation</p>
          <h1 className="mt-3 text-4xl font-semibold">Diarization accuracy</h1>
          <p className="mt-2 max-w-2xl text-sm text-zinc-500 dark:text-zinc-400">
            Runs this project&apos;s own WavLM + spectral clustering diarization pipeline on reference-annotated
            audio, then scores the generated hypothesis RTTM against the ground-truth RTTM.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="primary-btn" disabled={isRunning || !dataset?.files?.length} onClick={start}>
            {isRunning ? <Loader2 className="animate-spin" size={18} /> : <PlayCircle size={18} />}
            Run Evaluation
          </button>
          <button className="secondary-btn" disabled={isRunning} onClick={() => refreshAll()}>
            <RefreshCw size={18} /> Refresh Results
          </button>
          <button className="secondary-btn" disabled={isRunning} onClick={clearAll}>
            <Trash2 size={18} /> Clear Evaluation Results
          </button>
        </div>
      </div>

      {datasetError ? (
        <div className="glass-card flex items-start gap-3 p-5 text-sm">
          <AlertTriangle className="mt-0.5 shrink-0 text-amber-500" size={18} />
          <p className="text-zinc-600 dark:text-zinc-300">{datasetError}</p>
        </div>
      ) : null}

      <section className="glass-card p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-xl font-semibold">Evaluation Dataset</h2>
          {dataset ? (
            <p className="text-xs uppercase tracking-[0.18em] text-zinc-500 dark:text-zinc-400">
              {dataset.metrics_backend} &middot; collar {dataset.protocol.collar}s &middot;{" "}
              {dataset.protocol.skip_overlap ? "overlap skipped" : "overlap scored"} &middot;{" "}
              {dataset.protocol.speaker_count_mode} speaker count
            </p>
          ) : null}
        </div>
        {loading ? (
          <div className="skeleton-block mt-5 h-24" />
        ) : dataset?.files?.length ? (
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {dataset.files.map((file) => (
              <div
                key={file.file}
                className="rounded-lg border border-black/10 bg-white/50 p-4 dark:border-white/10 dark:bg-white/[0.04]"
              >
                <div className="flex items-center gap-2">
                  <FileAudio className="shrink-0 text-teal-400" size={18} />
                  <span className="min-w-0 truncate font-medium">{file.file}</span>
                </div>
                <dl className="mt-3 grid grid-cols-2 gap-y-1 text-xs text-zinc-500 dark:text-zinc-400">
                  <dt>Reference RTTM</dt>
                  <dd className="text-right text-zinc-700 dark:text-zinc-200">{file.name}.rttm</dd>
                  <dt>Recording ID</dt>
                  <dd className="text-right text-zinc-700 dark:text-zinc-200">{file.recording_id || "--"}</dd>
                  <dt>Duration</dt>
                  <dd className="text-right text-zinc-700 dark:text-zinc-200">
                    {file.duration ? `${file.duration.toFixed(2)}s` : "--"}
                  </dd>
                  <dt>Reference speakers</dt>
                  <dd className="text-right text-zinc-700 dark:text-zinc-200">{file.reference_speakers ?? "--"}</dd>
                  {file.lab_file ? (
                    <>
                      <dt>Optional .lab</dt>
                      <dd className="text-right text-zinc-700 dark:text-zinc-200">{file.lab_file}</dd>
                    </>
                  ) : null}
                </dl>
                {file.reference_error ? (
                  <p className="mt-3 rounded-lg bg-red-500/10 p-2 text-xs text-red-500 dark:text-red-300">
                    {file.reference_error}
                  </p>
                ) : null}
              </div>
            ))}
          </div>
        ) : (
          <div className="mt-5">
            <EmptyState
              title="No annotated evaluation files found"
              body="Place an audio file and a reference RTTM with the same name in the evaluation/ folder, for example LDC2021S10.flac and LDC2021S10.rttm."
            />
          </div>
        )}
        {dataset?.unpaired_audio?.length ? (
          <p className="mt-4 text-xs text-amber-600 dark:text-amber-400">
            Skipped without a reference RTTM: {dataset.unpaired_audio.join(", ")}
          </p>
        ) : null}
      </section>

      {status.status !== "idle" ? (
        <section className="glass-card p-6">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm text-zinc-500 dark:text-zinc-400">Current Stage</p>
              <h2 className="mt-1 text-xl font-semibold">{status.stage}</h2>
            </div>
            {isRunning ? (
              <Loader2 className="animate-spin text-teal-400" size={28} />
            ) : (
              <Gauge className="text-teal-400" size={28} />
            )}
          </div>
          <div className="mt-6 h-3 overflow-hidden rounded-full bg-zinc-200 dark:bg-white/10">
            <div
              className="h-full rounded-full bg-gradient-to-r from-teal-400 to-sky-400 transition-all duration-500"
              style={{ width: `${status.progress || 0}%` }}
            />
          </div>
          <div className="mt-2 flex justify-between text-xs text-zinc-500 dark:text-zinc-400">
            <span>{status.status}</span>
            <span>
              {status.progress || 0}%
              {status.processing_time ? ` · ${status.processing_time}s` : ""}
            </span>
          </div>
          {status.error ? (
            <p className="mt-4 rounded-lg bg-red-500/10 p-3 text-sm text-red-500 dark:text-red-300">{status.error}</p>
          ) : null}
          <h3 className="mt-6 text-sm font-semibold uppercase tracking-[0.18em] text-zinc-500">Realtime Logs</h3>
          <div className="mt-3 h-64 overflow-auto rounded-lg bg-zinc-950 p-4 font-mono text-xs leading-6 text-zinc-200">
            {logs.length ? (
              logs.map((line, index) => <p key={`${index}-${line}`}>{line}</p>)
            ) : (
              <p className="text-zinc-500">Logs will appear here once evaluation starts.</p>
            )}
          </div>
        </section>
      ) : null}

      {summary ? (
        <>
          <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Overall DER"
              value={formatPercent(summary.overall_der_percent)}
              detail="Duration-weighted across all scored files"
            />
            <MetricCard
              label="Overall JER"
              value={summary.jer_available ? formatPercent(summary.overall_jer_percent) : "Unavailable"}
              detail={summary.jer_available ? "Duration-weighted Jaccard error rate" : "JER not reported by the scoring backend"}
            />
            <MetricCard
              label="Files Evaluated"
              value={summary.file_count ?? "--"}
              detail={summary.total_duration ? `${summary.total_duration.toFixed(2)}s of audio` : ""}
            />
            <MetricCard
              label="Average Speakers"
              value={`${summary.average_reference_speakers ?? "--"} / ${summary.average_detected_speakers ?? "--"}`}
              detail="Reference / detected"
            />
            <MetricCard
              label="Per-file Mean DER"
              value={formatPercent(summary.average_der_percent)}
              detail="Unweighted mean of per-file rates"
            />
            <MetricCard
              label="Lowest DER"
              value={formatPercent(summary.min_der_percent)}
              detail={summary.min_der_file || ""}
            />
            <MetricCard
              label="Highest DER"
              value={formatPercent(summary.max_der_percent)}
              detail={summary.max_der_file || ""}
            />
            <MetricCard
              label="Total Error Time"
              value={`${(
                (summary.total_false_alarm || 0) +
                (summary.total_missed_speech || 0) +
                (summary.total_speaker_confusion || 0)
              ).toFixed(2)}s`}
              detail="False alarm + missed + confusion"
            />
          </section>

          <section className="grid gap-5 lg:grid-cols-2">
            <ChartCard title="DER and JER per File" hint="Lower is better">
              <BarChart data={derData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,.25)" />
                <XAxis dataKey="file" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" unit="%" />
                <Tooltip formatter={(value) => (value === null ? "n/a" : `${Number(value).toFixed(2)}%`)} />
                <Legend />
                <Bar dataKey="der" name="DER %" fill="#2dd4bf" radius={[6, 6, 0, 0]} />
                {summary.jer_available ? (
                  <Bar dataKey="jer" name="JER %" fill="#38bdf8" radius={[6, 6, 0, 0]} />
                ) : null}
              </BarChart>
            </ChartCard>

            <ChartCard title="Error Breakdown" hint="Seconds of each DER component">
              <BarChart data={breakdownData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,.25)" />
                <XAxis dataKey="file" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" unit="s" />
                <Tooltip formatter={(value) => `${Number(value).toFixed(2)}s`} />
                <Legend />
                <Bar dataKey="False Alarm" stackId="error" fill="#fbbf24" />
                <Bar dataKey="Missed Speech" stackId="error" fill="#f87171" />
                <Bar dataKey="Speaker Confusion" stackId="error" fill="#a78bfa" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ChartCard>

            <ChartCard title="Reference vs Detected Speakers" hint="Counts per file">
              <BarChart data={speakerData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,.25)" />
                <XAxis dataKey="file" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" allowDecimals={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="Reference" fill="#2dd4bf" radius={[6, 6, 0, 0]} />
                <Bar dataKey="Detected" fill="#f472b6" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ChartCard>

            <ChartCard title="Correct vs Error Time" hint="Share of the scored region">
              <BarChart
                data={records.map((record) => ({
                  file: record.name || record.file,
                  Correct: Math.max((record.scored_duration || 0) - (record.total_error || 0), 0),
                  Error: record.total_error || 0,
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,.25)" />
                <XAxis dataKey="file" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" unit="s" />
                <Tooltip formatter={(value) => `${Number(value).toFixed(2)}s`} />
                <Legend />
                <Bar dataKey="Correct" stackId="span" fill="#34d399" />
                <Bar dataKey="Error" stackId="span" fill="#fb7185" radius={[6, 6, 0, 0]}>
                  {records.map((record) => (
                    <Cell key={record.name || record.file} />
                  ))}
                </Bar>
              </BarChart>
            </ChartCard>
          </section>

          <section className="glass-card overflow-hidden p-0">
            <div className="flex flex-wrap items-center justify-between gap-3 p-6 pb-4">
              <h2 className="text-xl font-semibold">Per-file Results</h2>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Per-file DER, not the overall dataset DER. Overall DER is duration-weighted above.
              </p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[52rem] text-left text-sm">
                <thead className="border-y border-black/10 bg-white/40 text-xs uppercase tracking-[0.14em] text-zinc-500 dark:border-white/10 dark:bg-white/[0.04] dark:text-zinc-400">
                  <tr>
                    <SortableHeader column="file" label="File" sort={sort} onSort={toggleSort} />
                    <SortableHeader column="duration" label="Duration" sort={sort} onSort={toggleSort} />
                    <SortableHeader column="reference_speakers" label="Ref Speakers" sort={sort} onSort={toggleSort} />
                    <SortableHeader column="detected_speakers" label="Detected" sort={sort} onSort={toggleSort} />
                    <SortableHeader column="der" label="DER" sort={sort} onSort={toggleSort} />
                    <SortableHeader column="jer" label="JER" sort={sort} onSort={toggleSort} />
                    <th className="px-6 py-3">False Alarm</th>
                    <th className="px-6 py-3">Missed</th>
                    <th className="px-6 py-3">Confusion</th>
                    <th className="px-6 py-3">Files</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((record) => (
                    <tr
                      key={record.name || record.file}
                      className="border-b border-black/5 last:border-0 dark:border-white/5"
                    >
                      <td className="px-6 py-3 font-medium">{record.file}</td>
                      <td className="px-6 py-3">{record.duration ? `${record.duration.toFixed(2)}s` : "--"}</td>
                      <td className="px-6 py-3">{record.reference_speakers ?? "--"}</td>
                      <td className="px-6 py-3">{record.detected_speakers ?? "--"}</td>
                      <td className="px-6 py-3 font-semibold text-teal-600 dark:text-teal-300">
                        {formatPercent(record.der_percent)}
                      </td>
                      <td className="px-6 py-3">{formatPercent(record.jer_percent)}</td>
                      <td className="px-6 py-3">{formatSeconds(record.false_alarm)}</td>
                      <td className="px-6 py-3">{formatSeconds(record.missed_speech)}</td>
                      <td className="px-6 py-3">{formatSeconds(record.speaker_confusion)}</td>
                      <td className="px-6 py-3">
                        <div className="flex gap-1">
                          {downloads[record.name]?.evaluation ? (
                            <a
                              className="icon-btn"
                              href={evaluationDownloadUrl(downloads[record.name].evaluation)}
                              aria-label={`Download evaluation.json for ${record.file}`}
                              title="evaluation.json"
                            >
                              <Download size={16} />
                            </a>
                          ) : null}
                          {downloads[record.name]?.hypothesis ? (
                            <a
                              className="icon-btn"
                              href={evaluationDownloadUrl(downloads[record.name].hypothesis)}
                              aria-label={`Download hypothesis.rttm for ${record.file}`}
                              title="hypothesis.rttm"
                            >
                              <FileAudio size={16} />
                            </a>
                          ) : null}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3 border-t border-black/10 p-6 text-xs text-zinc-500 dark:border-white/10 dark:text-zinc-400">
              <span>
                Protocol: collar {summary.protocol?.collar}s,{" "}
                {summary.protocol?.skip_overlap ? "overlap skipped" : "overlap scored"}, UEM{" "}
                {summary.protocol?.uem_mode}, {summary.protocol?.speaker_count_mode} speaker count &middot; scored with{" "}
                {summary.metrics_backend}
                {summary.metrics_backend_version ? ` ${summary.metrics_backend_version}` : ""}
              </span>
              <a className="secondary-btn" href={evaluationDownloadUrl("summary.json")}>
                <Download size={16} /> summary.json
              </a>
            </div>
          </section>

          {summary.failures?.length ? (
            <section className="glass-card p-6">
              <h2 className="flex items-center gap-2 text-xl font-semibold">
                <AlertTriangle className="text-amber-500" size={20} /> Files that could not be evaluated
              </h2>
              <ul className="mt-4 space-y-2 text-sm">
                {summary.failures.map((failure) => (
                  <li key={failure.name} className="rounded-lg bg-amber-500/10 p-3">
                    <span className="font-medium">{failure.file}</span>
                    <span className="text-zinc-600 dark:text-zinc-300"> — {failure.error}</span>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
        </>
      ) : (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <EmptyState
            title="No evaluation results yet"
            body="Press Run Evaluation to diarize every annotated file and compute DER against the reference RTTM. No DER value is shown until it has actually been measured."
          />
        </motion.div>
      )}
    </div>
  );
}

function ChartCard({ title, hint, children }) {
  return (
    <article className="glass-card p-5">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-xl font-semibold">{title}</h2>
        {hint ? <span className="text-xs text-zinc-500 dark:text-zinc-400">{hint}</span> : null}
      </div>
      <div className="mt-4 h-72">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </article>
  );
}

function SortableHeader({ column, label, sort, onSort }) {
  const active = sort.column === column;
  return (
    <th className="px-6 py-3">
      <button
        type="button"
        className="flex items-center gap-1 uppercase tracking-[0.14em] transition hover:text-teal-500"
        onClick={() => onSort(column)}
      >
        {label}
        {active ? (
          sort.direction === "asc" ? (
            <ArrowUp size={12} />
          ) : (
            <ArrowDown size={12} />
          )
        ) : null}
      </button>
    </th>
  );
}

function formatPercent(value) {
  return value === null || value === undefined ? "--" : `${Number(value).toFixed(2)}%`;
}

function formatSeconds(value) {
  return value === null || value === undefined ? "--" : `${Number(value).toFixed(2)}s`;
}
