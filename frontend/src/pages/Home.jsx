import { useEffect, useState } from "react";
import { Loader2, Music, ThumbsUp, ThumbsDown } from "lucide-react";
import SongUploader from "../components/mashup/SongUploader";
import ResultCard from "../components/mashup/ResultCard";
import { analyzeSongs } from "../api/analyzeClient";

const VinylRecord = () => (
  <div className="relative w-96 h-96 shrink-0">
    {/* Outer disc (Spinning) */}
    <div className="absolute inset-0 rounded-full bg-[#111] border-[5px] border-foreground flex items-center justify-center overflow-hidden animate-spin-slow">
      {/* Grooves / Refraction */}
      <div className="absolute inset-0 rounded-full opacity-30" style={{
        background: 'conic-gradient(from 0deg, transparent 0deg, rgba(255,255,255,0.4) 45deg, transparent 90deg, rgba(255,255,255,0.4) 135deg, transparent 180deg, rgba(255,255,255,0.4) 225deg, transparent 270deg, rgba(255,255,255,0.4) 315deg, transparent 360deg)'
      }} />
      <div className="absolute inset-4 rounded-full border border-white/10" />
      <div className="absolute inset-8 rounded-full border border-white/5" />
      <div className="absolute inset-14 rounded-full border border-white/10" />
      <div className="absolute inset-20 rounded-full border border-white/5" />
      <div className="absolute inset-24 rounded-full border border-white/10" />
      <div className="absolute inset-32 rounded-full border border-white/5" />
      <div className="absolute inset-36 rounded-full border border-white/10" />
      
      {/* Track breaks (asymmetrical elements to show spinning) */}
      <div className="absolute w-[90%] h-[90%] border-[5px] border-transparent border-t-[#333] rounded-full opacity-70 rotate-45" />
      <div className="absolute w-[75%] h-[75%] border-4 border-transparent border-b-[#444] rounded-full opacity-50 -rotate-12" />
      
      {/* Label */}
      <div className="w-32 h-32 rounded-full bg-primary border-[5px] border-foreground flex items-center justify-center relative overflow-hidden">
        {/* Label details to make spin obvious */}
        <div className="absolute inset-0 bg-white/20 w-full h-1/2 top-0" />
        <div className="absolute w-full h-2 bg-foreground/20 rotate-45" />
        <div className="absolute text-base font-display tracking-widest text-background font-bold top-5">MASH</div>
        <div className="absolute text-base font-display tracking-widest text-background font-bold bottom-5">LAB</div>
        <div className="w-6 h-6 rounded-full bg-background border-2 border-foreground z-10" />
      </div>
    </div>
  </div>
);

export default function Home() {
  const [song1, setSong1] = useState(null);
  const [song2, setSong2] = useState(null);
  const [song1Name, setSong1Name] = useState("");
  const [song2Name, setSong2Name] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    return () => {
      if (result?.preview_url?.startsWith("blob:")) {
        URL.revokeObjectURL(result.preview_url);
      }
    };
  }, [result]);

  const canAnalyze = Boolean(song1 && song2 && !isAnalyzing);

  const handleAnalyze = async () => {
    if (!canAnalyze) return;

    setErrorMessage("");
    setResult(null);
    setIsAnalyzing(true);

    try {
      const backendResult = await analyzeSongs(song1, song2);
      setResult(backendResult);
    } catch (error) {
      console.error(error);
      setErrorMessage(error.message || "Unable to analyze tracks right now.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSong1(null);
    setSong2(null);
    setSong1Name("");
    setSong2Name("");
    setResult(null);
    setErrorMessage("");
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Top Border */}
      <div className="w-full h-4 bg-foreground border-b-2 border-foreground" />

      <div className="flex-1 flex flex-col md:flex-row">
        {/* LEFT: Big title column */}
        <div className="md:w-2/5 border-r-3 border-foreground flex flex-col justify-between p-10 md:p-16 min-h-[50vh]">
          <div>
            <div className="w-10 h-10 rounded-full bg-primary border-3 border-foreground mb-10" />
            <h1 className="font-display text-[clamp(5rem,12vw,9rem)] leading-none text-foreground">
              MASH
              <br />
              <span className="text-primary">LAB</span>
            </h1>
            <p className="font-mono text-lg text-muted-foreground mt-10 max-w-sm leading-relaxed">
              Drop two tracks. We&apos;ll tell you if they belong together — and let you hear it.
            </p>
          </div>

          {/* Spinning Vinyl */}
          <div className="flex-1 flex items-center justify-center w-full mt-12 pb-8">
            <VinylRecord />
          </div>
        </div>

        {/* RIGHT: Interaction column */}
        <div className="md:w-3/5 flex flex-col p-10 md:p-16 gap-12">
          {!result ? (
            <>
              {/* Step label */}
              <div className="flex items-center gap-4">
                <span className="font-mono text-base text-muted-foreground tracking-widest uppercase font-bold">
                  01 - Track Information
                </span>
                <div className="flex-1 h-px bg-border" />
              </div>

              {/* Spotify-style Inputs */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-10">
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-2">
                    <label className="font-mono text-sm uppercase tracking-wider text-muted-foreground">Track A Name (Spotify)</label>
                    <input 
                      type="text" 
                      value={song1Name}
                      onChange={(e) => setSong1Name(e.target.value)}
                      placeholder="e.g. Blinding Lights"
                      className="brutal-card p-4 font-mono text-base bg-white focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>
                  <SongUploader
                    label="TRACK A"
                    file={song1}
                    onFileChange={setSong1}
                    disabled={isAnalyzing}
                    index="A"
                  />
                </div>
                
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-2">
                    <label className="font-mono text-sm uppercase tracking-wider text-muted-foreground">Track B Name (Spotify)</label>
                    <input 
                      type="text" 
                      value={song2Name}
                      onChange={(e) => setSong2Name(e.target.value)}
                      placeholder="e.g. Save Your Tears"
                      className="brutal-card p-4 font-mono text-base bg-white focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>
                  <SongUploader
                    label="TRACK B"
                    file={song2}
                    onFileChange={setSong2}
                    disabled={isAnalyzing}
                    index="B"
                  />
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="flex-1 h-px bg-border" />
                <span className="font-mono text-base text-muted-foreground tracking-widest uppercase font-bold">
                  02 - Analyze
                </span>
                <div className="flex-1 h-px bg-border" />
              </div>

              <button
                onClick={handleAnalyze}
                disabled={!canAnalyze}
                className="brutal-btn w-full py-8 text-2xl tracking-widest font-display uppercase flex items-center justify-center gap-4"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-6 h-6 animate-spin" />
                    ANALYZING...
                  </>
                ) : (
                  "RUN COMPATIBILITY CHECK ->"
                )}
              </button>

              {!canAnalyze && !isAnalyzing && (
                <p className="font-mono text-base text-muted-foreground text-center -mt-8">
                  {!song1 && !song2
                    ? "Upload both tracks to continue"
                    : !song1
                      ? "Still need Track A"
                      : "Still need Track B"}
                </p>
              )}

              {errorMessage && (
                <p className="font-mono text-base text-destructive text-center">{errorMessage}</p>
              )}
            </>
          ) : (
            <>
              <div className="flex items-center gap-4">
                <span className="font-mono text-base text-muted-foreground tracking-widest uppercase font-bold">
                  03 - Results
                </span>
                <div className="flex-1 h-px bg-border" />
              </div>

              <ResultCard
                result={result}
                song1Name={song1Name || song1?.name}
                song2Name={song2Name || song2?.name}
                sessionId={result.session_id}
              />

              <button
                onClick={handleReset}
                className="font-mono text-base text-muted-foreground hover:text-foreground transition-colors self-start border-b border-muted-foreground hover:border-foreground pb-1 mt-6"
              >
                {"<- try different tracks"}
              </button>
            </>
          )}
        </div>
      </div>

      <div className="border-t-3 border-foreground px-10 py-6 flex justify-between items-center">
        <span className="font-mono text-base text-muted-foreground">MASHLAB v1.0</span>
      </div>
    </div>
  );
}
