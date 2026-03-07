import { useRef, useState } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

export default function SongUploader({ label, file, onFileChange, disabled, index }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && (dropped.type === "audio/mpeg" || dropped.name.endsWith(".mp3"))) onFileChange(dropped);
  };

  const handleFileInput = (e) => {
    const selected = e.target.files[0];
    if (selected) onFileChange(selected);
  };

  return (
    <div
      className={cn(
        "brutal-card p-5 flex flex-col gap-4 cursor-pointer relative min-h-[180px] select-none",
        dragging && "brutal-card-active bg-primary/10",
        disabled && "opacity-40 pointer-events-none"
      )}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/mpeg,audio/mp3,.mp3"
        className="hidden"
        onChange={handleFileInput}
      />

      {/* Index badge */}
      <div className="flex items-center justify-between">
        <span className="font-display text-5xl leading-none text-border select-none">{index}</span>
        {file && (
          <button
            className="w-6 h-6 border-2 border-foreground flex items-center justify-center hover:bg-primary hover:border-primary hover:text-white transition-colors"
            onClick={(e) => { e.stopPropagation(); onFileChange(null); }}
          >
            <X className="w-3 h-3" />
          </button>
        )}
      </div>

      <div className="mt-auto">
        <p className="font-display text-lg text-foreground">{label}</p>
        {file ? (
          <>
            <div className="h-px bg-primary my-2" />
            <p className="font-mono text-xs text-foreground truncate">{file.name}</p>
            <p className="font-mono text-xs text-muted-foreground mt-0.5">{(file.size / 1024 / 1024).toFixed(2)} MB · MP3</p>
          </>
        ) : (
          <>
            <div className="h-px bg-border my-2" />
            <p className="font-mono text-xs text-muted-foreground">drag & drop or click</p>
            <p className="font-mono text-xs text-muted-foreground">.mp3 only</p>
          </>
        )}
      </div>

      {/* Corner accent */}
      {file && (
        <div className="absolute bottom-0 right-0 w-0 h-0"
          style={{
            borderLeft: "20px solid transparent",
            borderBottom: "20px solid hsl(var(--primary))",
          }}
        />
      )}
    </div>
  );
}