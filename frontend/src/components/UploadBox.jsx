import { useCallback, useRef, useState } from "react";

const ACCEPT = "image/jpeg,image/png,image/webp,image/bmp,video/mp4,video/quicktime,video/webm,.mkv,.avi";

export default function UploadBox({ file, previewUrl, onFileSelected, disabled }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const pickFiles = useCallback(
    (files) => {
      if (!files?.length || disabled) return;
      onFileSelected(files[0]);
    },
    [disabled, onFileSelected]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      pickFiles(e.dataTransfer.files);
    },
    [pickFiles]
  );

  const onDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setDragOver(true);
  };

  const onDragLeave = () => setDragOver(false);

  return (
    <div className="space-y-4">
      <div
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onClick={() => !disabled && inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        className={`
          relative rounded-2xl border-2 border-dashed px-6 py-12 text-center cursor-pointer
          transition-colors
          ${dragOver ? "border-emerald-400 bg-emerald-500/10" : "border-slate-600 bg-slate-900/50"}
          ${disabled ? "opacity-50 pointer-events-none" : "hover:border-slate-500"}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          disabled={disabled}
          onChange={(e) => pickFiles(e.target.files)}
        />
        <p className="font-display text-lg font-semibold text-slate-200 mb-1">
          Drop file here or click to browse
        </p>
        <p className="text-sm text-slate-500">Images (JPG, PNG, …) or video (MP4, MOV, …)</p>
      </div>

      {previewUrl && file && (
        <div className="rounded-xl overflow-hidden border border-slate-700 bg-slate-900">
          {file.type.startsWith("video/") ? (
            <video src={previewUrl} controls className="w-full max-h-64 object-contain bg-black" />
          ) : (
            <img src={previewUrl} alt="Preview" className="w-full max-h-64 object-contain" />
          )}
          <p className="px-3 py-2 text-xs text-slate-500 truncate">{file.name}</p>
        </div>
      )}
    </div>
  );
}
