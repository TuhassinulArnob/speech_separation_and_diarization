import { AudioWaveform } from "lucide-react";

export default function EmptyState({ title, body }) {
  return (
    <div className="grid min-h-72 place-items-center rounded-xl border border-dashed border-black/15 bg-white/45 p-8 text-center dark:border-white/15 dark:bg-white/[0.03]">
      <div>
        <AudioWaveform className="mx-auto text-teal-400" size={42} />
        <h2 className="mt-4 text-xl font-semibold">{title}</h2>
        <p className="mx-auto mt-2 max-w-md text-sm text-zinc-500 dark:text-zinc-400">{body}</p>
      </div>
    </div>
  );
}
