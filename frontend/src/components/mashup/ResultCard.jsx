import { useState } from "react";
import { ThumbsUp, ThumbsDown, Check } from "lucide-react";
import CompatibilityMeter from "./CompatibilityMeter";
import MashupPlayer from "./MashupPlayer";
import { cn } from "@/lib/utils";

// ── Backend integration point ──────────────────────────────────────────────
const COMPATIBILITY_THRESHOLD = 60;
// ──────────────────────────────────────────────────────────────────────────

export default function ResultCard({ result, song1Name, song2Name }) {
  const isCompatible = result.score >= COMPATIBILITY_THRESHOLD;
  const [rating, setRating] = useState(null); // 'up', 'down', or null

  return (
    <div className="w-full flex flex-col gap-10 animate-fade-up">
      {/* Track names */}
      <div className="flex items-center gap-5 font-mono text-base text-muted-foreground font-bold">
        <span className="truncate max-w-[180px]">{song1Name?.replace(/\.[^/.]+$/, "") || "Track A"}</span>
        <span className="text-primary font-bold text-3xl">×</span>
        <span className="truncate max-w-[180px]">{song2Name?.replace(/\.[^/.]+$/, "") || "Track B"}</span>
      </div>

      <CompatibilityMeter score={result.score} />

      <div className="h-px bg-foreground" />

      {isCompatible ? (
        <div className="flex flex-col gap-8">
          <MashupPlayer previewUrl={result.preview_url} songName={result.preview_song_name} />
          
          {/* Rating System */}
          <div className="flex flex-col items-center gap-6 p-8 brutal-card bg-secondary/30">
            <p className="font-display text-2xl uppercase tracking-widest">Rate this Mashup</p>
            <div className="flex gap-6">
              <button
                onClick={() => setRating('up')}
                className={cn(
                  "w-20 h-20 border-3 border-foreground flex items-center justify-center transition-all",
                  rating === 'up' ? "bg-primary text-white scale-110 shadow-none translate-x-1 translate-y-1" : "bg-white hover:bg-primary/10 brutal-btn-shadow"
                )}
                style={{
                  boxShadow: rating === 'up' ? "none" : "6px 6px 0px hsl(var(--foreground))"
                }}
              >
                <ThumbsUp className={cn("w-10 h-10", rating === 'up' && "animate-bounce")} />
              </button>
              
              <button
                onClick={() => setRating('down')}
                className={cn(
                  "w-20 h-20 border-3 border-foreground flex items-center justify-center transition-all",
                  rating === 'down' ? "bg-foreground text-white scale-110 shadow-none translate-x-1 translate-y-1" : "bg-white hover:bg-foreground/10 brutal-btn-shadow"
                )}
                style={{
                  boxShadow: rating === 'down' ? "none" : "6px 6px 0px hsl(var(--foreground))"
                }}
              >
                <ThumbsDown className={cn("w-10 h-10", rating === 'down' && "animate-bounce")} />
              </button>
            </div>
            {rating && (
              <p className="font-mono text-base font-bold text-primary flex items-center gap-2 animate-fade-up">
                <Check className="w-5 h-5" /> Thanks for your feedback!
              </p>
            )}
          </div>
        </div>
      ) : (
        <div className="brutal-card p-10 flex flex-col gap-5">
          <p className="font-display text-5xl text-foreground">NOT COMPATIBLE</p>
          <div className="h-px bg-border" />
          <p className="font-mono text-base text-muted-foreground leading-relaxed font-bold">
            Score below {COMPATIBILITY_THRESHOLD}. Try tracks with similar keys or BPMs for a cleaner mash.
          </p>
        </div>
      )}
    </div>
  );
}