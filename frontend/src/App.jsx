import { useCallback, useState } from "react";
import UploadBox from "./components/UploadBox.jsx";
import Loader from "./components/Loader.jsx";
import ResultCard from "./components/ResultCard.jsx";

/**
 * Backend URL: Flask serves POST /predict on port 5000.
 * The browser sends multipart/form-data with field name "file" (see handleAnalyze).
 */
const API_BASE = "http://127.0.0.1:5000";

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [confidence, setConfidence] = useState(null);

  const onFileSelected = useCallback((selected) => {
    setFile(selected);
    setError(null);
    setResult(null);
    setConfidence(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return selected ? URL.createObjectURL(selected) : null;
    });
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setConfidence(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.error || `Request failed (${res.status})`);
      }
      setResult(data.result);
      setConfidence(typeof data.confidence === "number" ? data.confidence : null);
    } catch (e) {
      setError(e.message || "Network error — is the Flask server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-12">
      <header className="text-center mb-10 max-w-xl">
        <h1 className="font-display text-3xl sm:text-4xl font-bold tracking-tight text-white mb-2">
          DeepTrace
        </h1>
        <p className="text-slate-400 text-sm sm:text-base">
          Hybrid spatial–frequency deepfake detection. Upload an image or video frame for analysis.
        </p>
      </header>

      <main className="w-full max-w-lg space-y-8">
        <UploadBox
          file={file}
          previewUrl={previewUrl}
          onFileSelected={onFileSelected}
          disabled={loading}
        />

        <div className="flex justify-center">
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="px-8 py-3 rounded-xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 disabled:cursor-not-allowed text-slate-950 font-semibold transition-colors shadow-lg shadow-emerald-500/20"
          >
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>

        {loading && <Loader />}

        {error && (
          <div
            className="rounded-xl border border-red-500/40 bg-red-950/50 px-4 py-3 text-sm text-red-200"
            role="alert"
          >
            {error}
          </div>
        )}

        {result != null && confidence != null && !loading && (
          <ResultCard result={result} confidence={confidence} />
        )}
      </main>

      <footer className="mt-16 text-center text-xs text-slate-600">
        API: <code className="text-slate-500">{API_BASE}/predict</code>
      </footer>
    </div>
  );
}
