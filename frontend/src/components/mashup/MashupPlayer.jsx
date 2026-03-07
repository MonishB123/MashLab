import { useEffect, useRef, useState } from "react";
import { Play, Pause } from "lucide-react";

export default function MashupPlayer({ previewUrl, songName }) {
  const audioRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onTime = () => setCurrentTime(audio.currentTime);
    const onLoad = () => setDuration(audio.duration);
    const onEnd = () => setPlaying(false);
    audio.addEventListener("timeupdate", onTime);
    audio.addEventListener("loadedmetadata", onLoad);
    audio.addEventListener("ended", onEnd);
    return () => {
      audio.removeEventListener("timeupdate", onTime);
      audio.removeEventListener("loadedmetadata", onLoad);
      audio.removeEventListener("ended", onEnd);
    };
  }, []);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    playing ? audio.pause() : audio.play();
    setPlaying(!playing);
  };

  const handleSeek = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    const newTime = ratio * (duration || 0);
    if (audioRef.current) audioRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const fmt = (s) => `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, "0")}`;
  const progress = duration ? currentTime / duration : 0;

  return (
    <div className="brutal-card p-8 animate-fade-up w-full">
      <audio ref={audioRef} src={previewUrl} />

      <div className="flex items-center justify-between mb-8">
        <div>
          <p className="font-mono text-base text-muted-foreground tracking-widest uppercase mb-1 font-bold">Preview Clip</p>
          <p className="font-display text-4xl text-foreground truncate max-w-[320px]">
            {songName || "MASHUP PREVIEW"}
          </p>
        </div>
        <button
          onClick={togglePlay}
          className="w-16 h-16 border-3 border-foreground bg-foreground text-background flex items-center justify-center hover:bg-primary hover:border-primary transition-colors"
        >
          {playing
            ? <Pause className="w-8 h-8" />
            : <Play className="w-8 h-8 ml-1" />
          }
        </button>
      </div>

      {/* Waveform-style progress bar — clickable */}
      <div
        className="w-full h-16 border-3 border-foreground cursor-pointer relative overflow-hidden"
        onClick={handleSeek}
      >
        {/* Fill */}
        <div
          className="absolute inset-0 bg-primary transition-none origin-left"
          style={{ transform: `scaleX(${progress})` }}
        />
        {/* Fake waveform bars */}
        <div className="absolute inset-0 flex items-center gap-px px-2 pointer-events-none">
          {[...Array(48)].map((_, i) => {
            const h = 25 + Math.sin(i * 0.8) * 15 + Math.cos(i * 1.3) * 10;
            const filled = (i / 48) < progress;
            return (
              <div
                key={i}
                className="flex-1"
                style={{
                  height: `${h}%`,
                  background: filled ? "hsl(var(--background))" : "hsl(var(--foreground))",
                  opacity: filled ? 0.9 : 0.4,
                }}
              />
            );
          })}
        </div>
      </div>

      <div className="flex justify-between mt-3 font-bold">
        <span className="font-mono text-base text-muted-foreground">{fmt(currentTime)}</span>
        <span className="font-mono text-base text-muted-foreground">{fmt(duration)}</span>
      </div>
    </div>
  );
}