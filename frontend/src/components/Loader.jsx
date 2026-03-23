export default function Loader() {
  return (
    <div className="flex flex-col items-center gap-3 py-4" aria-live="polite" aria-busy="true">
      <div className="h-10 w-10 rounded-full border-2 border-slate-600 border-t-emerald-400 animate-spin" />
      <p className="text-sm text-slate-400">Running model inference…</p>
    </div>
  );
}
