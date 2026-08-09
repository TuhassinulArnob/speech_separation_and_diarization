import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { Pause, Play } from "lucide-react";

export default function WaveformPlayer({ src, compact = false }) {
  const containerRef = useRef(null);
  const waveRef = useRef(null);
  const [isReady, setReady] = useState(false);
  const [isPlaying, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!containerRef.current || !src) return undefined;
    const wave = WaveSurfer.create({
      container: containerRef.current,
      url: src,
      height: compact ? 54 : 86,
      waveColor: "rgba(148, 163, 184, 0.45)",
      progressColor: "#2dd4bf",
      cursorColor: "#f8fafc",
      barWidth: 2,
      barRadius: 2,
      barGap: 2,
      normalize: true,
    });
    waveRef.current = wave;
    wave.on("ready", () => {
      setReady(true);
      setDuration(wave.getDuration());
    });
    wave.on("play", () => setPlaying(true));
    wave.on("pause", () => setPlaying(false));
    wave.on("finish", () => setPlaying(false));
    return () => wave.destroy();
  }, [src, compact]);

  return (
    <div className="rounded-lg border border-black/10 bg-white/55 p-3 dark:border-white/10 dark:bg-white/[0.04]">
      <div className="flex items-center gap-3">
        <button className="icon-btn shrink-0" disabled={!isReady} onClick={() => waveRef.current?.playPause()} aria-label={isPlaying ? "Pause" : "Play"}>
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <div ref={containerRef} className="min-w-0 flex-1" />
        <span className="w-16 text-right text-xs text-zinc-500 dark:text-zinc-400">{duration ? `${duration.toFixed(1)}s` : "--"}</span>
      </div>
    </div>
  );
}
