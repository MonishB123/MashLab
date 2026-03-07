import { useEffect, useState } from "react";

function getScoreColor(score) {
  if (score >= 70) return "hsl(var(--score-high))";
  if (score >= 40) return "hsl(var(--score-mid))";
  return "hsl(var(--score-low))";
}

function getScoreLabel(score) {
  if (score >= 70) return "FIRE COMBO";
  if (score >= 40) return "COULD WORK";
  return "NO DICE";
}

function getScoreSubtext(score) {
  if (score >= 70) return "These tracks were made for each other.";
  if (score >= 40) return "With the right edit, this could slap.";
  return "Stick to your day job... just kidding. Try again.";
}

export default function CompatibilityMeter({ score }) {
  const [displayScore, setDisplayScore] = useState(0);
  const color = getScoreColor(score);

  useEffect(() => {
    let start = 0;
    const step = score / (1400 / 16);
    const interval = setInterval(() => {
      start += step;
      if (start >= score) { start = score; clearInterval(interval); }
      setDisplayScore(Math.round(start));
    }, 16);
    return () => clearInterval(interval);
  }, [score]);

  // Bar segments (20 total)
  const TOTAL = 20;
  const filled = Math.round((displayScore / 100) * TOTAL);

  return (
    <div className="w-full animate-fade-up">
      {/* Score number */}
      <div className="flex items-end gap-6 mb-6">
        <span className="font-display leading-none" style={{ fontSize: "clamp(5rem, 12vw, 9rem)", color }}>
          {displayScore}
        </span>
        <div className="pb-4">
          <span className="font-display text-4xl text-muted-foreground">/100</span>
        </div>
        <div className="pb-4 ml-auto text-right">
          <p className="font-display text-5xl text-foreground">{getScoreLabel(score)}</p>
          <p className="font-mono text-base text-muted-foreground mt-2 max-w-[280px] font-bold">{getScoreSubtext(score)}</p>
        </div>
      </div>

      {/* Segmented bar */}
      <div className="flex gap-2">
        {[...Array(TOTAL)].map((_, i) => (
          <div
            key={i}
            className="flex-1 h-10 border-3 border-foreground transition-all duration-75"
            style={{
              background: i < filled ? color : "transparent",
              transitionDelay: `${i * 30}ms`,
            }}
          />
        ))}
      </div>

      <div className="flex justify-between mt-3 font-bold">
        <span className="font-mono text-base text-muted-foreground">0</span>
        <span className="font-mono text-base text-muted-foreground uppercase tracking-widest">COMPATIBILITY</span>
        <span className="font-mono text-base text-muted-foreground">100</span>
      </div>
    </div>
  );
}