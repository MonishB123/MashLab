import CompatibilityMeter from "./CompatibilityMeter";
import MashupPlayer from "./MashupPlayer";

// ── Backend integration point ──────────────────────────────────────────────
const COMPATIBILITY_THRESHOLD = 60;
// ──────────────────────────────────────────────────────────────────────────

export default function ResultCard({ result, song1Name, song2Name }) {
  const isCompatible = result.score >= COMPATIBILITY_THRESHOLD;

  return (
    <div className="w-full flex flex-col gap-8 animate-fade-up">
      {/* Track names */}
      <div className="flex items-center gap-3 font-mono text-xs text-muted-foreground">
        <span className="truncate max-w-[120px]">{song1Name?.replace(/\.[^/.]+$/, "") || "Track A"}</span>
        <span className="text-primary font-bold text-lg">×</span>
        <span className="truncate max-w-[120px]">{song2Name?.replace(/\.[^/.]+$/, "") || "Track B"}</span>
      </div>

      <CompatibilityMeter score={result.score} />

      <div className="h-px bg-foreground" />

      {isCompatible ? (
        <MashupPlayer previewUrl={result.preview_url} songName={result.preview_song_name} />
      ) : (
        <div className="brutal-card p-6 flex flex-col gap-3">
          <p className="font-display text-3xl text-foreground">NOT COMPATIBLE</p>
          <div className="h-px bg-border" />
          <p className="font-mono text-xs text-muted-foreground leading-relaxed">
            Score below {COMPATIBILITY_THRESHOLD}. Try tracks with similar keys or BPMs for a cleaner mash.
          </p>
        </div>
      )}
    </div>
  );
}