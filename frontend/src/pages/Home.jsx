import { motion } from "framer-motion";
import { AudioLines, BarChart3, BrainCircuit, FileCode2, GitBranch, Github, Mic2, UploadCloud } from "lucide-react";
import { Link } from "react-router-dom";

const features = [
  { icon: AudioLines, title: "Speech Separation", body: "SepFormer-powered source separation for overlapping speech." },
  { icon: Mic2, title: "Speaker Diarization", body: "Detects who spoke when across the full recording." },
  { icon: BrainCircuit, title: "WavLM Embeddings", body: "Self-supervised speaker representations extracted per segment." },
  { icon: GitBranch, title: "Spectral Clustering", body: "Groups normalized embeddings into consistent speaker tracks." },
  { icon: FileCode2, title: "RTTM Generation", body: "Exports standard diarization files for downstream evaluation." },
  { icon: BarChart3, title: "DER/JER Evaluation", body: "Summarizes model quality when reference annotations are available." },
];

export default function Home() {
  return (
    <div>
      <section className="relative overflow-hidden">
        <div className="mx-auto grid min-h-[calc(100vh-4rem)] max-w-7xl items-center gap-10 px-4 py-16 sm:px-6 lg:grid-cols-[1fr_.9fr]">
          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
            <p className="text-sm font-semibold uppercase tracking-[0.22em] text-teal-500 dark:text-teal-300">AI audio intelligence</p>
            <h1 className="mt-5 max-w-4xl text-5xl font-semibold leading-tight text-zinc-950 dark:text-white sm:text-6xl lg:text-7xl">
              Speech Separation & Speaker Diarization
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-zinc-600 dark:text-zinc-300">
              AI-powered system for separating overlapping speech and identifying who spoke when.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link className="primary-btn" to="/process"><UploadCloud size={18} /> Upload Audio</Link>
              <Link className="secondary-btn" to="/process">Try Demo</Link>
              <a className="secondary-btn" href="https://github.com/" target="_blank" rel="noreferrer"><Github size={18} /> GitHub Repository</a>
              <Link className="secondary-btn" to="/about">Documentation</Link>
            </div>
          </motion.div>
          <motion.div className="glass-card p-5" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.7, delay: 0.1 }}>
            <div className="rounded-lg bg-zinc-950 p-5 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-zinc-400">Live waveform</p>
                  <p className="mt-1 text-lg font-semibold">Separation pipeline</p>
                </div>
                <span className="rounded-full bg-teal-400/15 px-3 py-1 text-xs font-medium text-teal-200">Ready</span>
              </div>
              <div className="wave-bars mt-8 flex h-44 items-end gap-1.5">
                {Array.from({ length: 72 }).map((_, index) => (
                  <span
                    key={index}
                    className="w-full rounded-t bg-gradient-to-t from-teal-500 to-sky-300"
                    style={{ height: `${18 + ((index * 17) % 76)}%` }}
                  />
                ))}
              </div>
              <div className="mt-7 grid grid-cols-3 gap-3 text-center text-sm">
                <div className="rounded-lg bg-white/8 p-3"><p className="text-zinc-400">Stages</p><p className="mt-1 font-semibold">9</p></div>
                <div className="rounded-lg bg-white/8 p-3"><p className="text-zinc-400">Outputs</p><p className="mt-1 font-semibold">10+</p></div>
                <div className="rounded-lg bg-white/8 p-3"><p className="text-zinc-400">Tests</p><p className="mt-1 font-semibold">106</p></div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      <section className="mx-auto max-w-7xl px-4 pb-20 sm:px-6">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => (
            <motion.article key={feature.title} className="glass-card p-5" initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.04 }}>
              <feature.icon className="text-teal-400" size={28} />
              <h2 className="mt-4 text-lg font-semibold">{feature.title}</h2>
              <p className="mt-2 text-sm leading-6 text-zinc-600 dark:text-zinc-400">{feature.body}</p>
            </motion.article>
          ))}
        </div>
      </section>
    </div>
  );
}
