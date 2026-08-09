export default function MetricCard({ label, value, detail }) {
  return (
    <div className="glass-card p-5">
      <p className="text-sm text-zinc-500 dark:text-zinc-400">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-zinc-950 dark:text-white">{value ?? "--"}</p>
      {detail ? <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">{detail}</p> : null}
    </div>
  );
}
