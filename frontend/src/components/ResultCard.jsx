export default function ResultCard({ result, confidence }) {
  const isFake = result === "fake";
  const pct = Math.round(confidence * 1000) / 10;

  return (
    <div
      className={`
        rounded-2xl border px-6 py-5 shadow-xl
        ${isFake ? "border-red-500/50 bg-red-950/40" : "border-emerald-500/50 bg-emerald-950/40"}
      `}
    >
      <p className="text-xs uppercase tracking-wider text-slate-500 mb-1">Verdict</p>
      <p
        className={`font-display text-3xl font-bold ${isFake ? "text-red-400" : "text-emerald-400"}`}
      >
        {isFake ? "Fake" : "Real"}
      </p>
      <p className="mt-3 text-sm text-slate-400">
        Confidence:{" "}
        <span className="text-slate-200 font-semibold tabular-nums">{pct}%</span>
      </p>
    </div>
  );
}
