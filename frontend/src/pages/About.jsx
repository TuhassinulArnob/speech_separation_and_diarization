import { Cpu, Database, Layers3, Route, ServerCog } from "lucide-react";

const pipeline = ["Audio Preprocessing", "Voice Activity Detection", "Speech Segmentation", "WavLM Embedding Extraction", "Embedding Normalization", "Spectral Clustering", "Timeline Reconstruction", "RTTM Generation", "DER/JER Evaluation"];

export default function About() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6">
      <div className="max-w-3xl">
        <p className="text-sm font-semibold uppercase tracking-[0.22em] text-teal-500">Project overview</p>
        <h1 className="mt-4 text-4xl font-semibold sm:text-5xl">A production interface for a completed diarization engine.</h1>
        <p className="mt-5 text-lg leading-8 text-zinc-600 dark:text-zinc-300">
          The backend pipeline separates overlapping speech, detects speech regions, extracts WavLM speaker embeddings, clusters them, reconstructs speaker timelines, and exports evaluation-ready artifacts.
        </p>
      </div>
      <div className="mt-10 grid gap-5 lg:grid-cols-3">
        <Info icon={Layers3} title="Speech Separation" body="The app calls the existing SpeechBrain SepFormer wrapper to create speaker-specific audio tracks from a mixed recording." />
        <Info icon={Route} title="Speaker Diarization" body="The diarization runner keeps the established preprocessing, VAD, segmentation, embedding, clustering, and timeline flow intact." />
        <Info icon={ServerCog} title="System Architecture" body="React handles uploads, playback, logs, and visualization. FastAPI owns orchestration and file delivery." />
      </div>
      <div className="mt-8 grid gap-5 lg:grid-cols-[.85fr_1.15fr]">
        <section className="glass-card p-6">
          <Cpu className="text-teal-400" />
          <h2 className="mt-4 text-xl font-semibold">Technologies</h2>
          <div className="mt-4 flex flex-wrap gap-2">
            {["React 19", "Vite", "Tailwind CSS", "Framer Motion", "FastAPI", "WaveSurfer.js", "Recharts", "SpeechBrain", "WavLM", "PyTorch"].map((tech) => (
              <span key={tech} className="rounded-full border border-black/10 px-3 py-1 text-sm text-zinc-600 dark:border-white/10 dark:text-zinc-300">{tech}</span>
            ))}
          </div>
        </section>
        <section className="glass-card p-6">
          <Database className="text-teal-400" />
          <h2 className="mt-4 text-xl font-semibold">Pipeline</h2>
          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            {pipeline.map((stage, index) => (
              <div key={stage} className="flex items-center gap-3 rounded-lg border border-black/10 bg-white/50 p-3 dark:border-white/10 dark:bg-white/[0.04]">
                <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-teal-400/15 text-sm font-semibold text-teal-500 dark:text-teal-300">{index + 1}</span>
                <span className="text-sm font-medium">{stage}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

function Info({ icon: Icon, title, body }) {
  return (
    <article className="glass-card p-6">
      <Icon className="text-teal-400" />
      <h2 className="mt-4 text-xl font-semibold">{title}</h2>
      <p className="mt-3 text-sm leading-6 text-zinc-600 dark:text-zinc-400">{body}</p>
    </article>
  );
}
