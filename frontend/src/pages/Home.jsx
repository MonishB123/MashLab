import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import SongUploader from "../components/mashup/SongUploader";
import ResultCard from "../components/mashup/ResultCard";
import { analyzeSongs } from "../api/analyzeClient";

const MARQUEE_TEXT = "MASHLAB - AUDIO COMPATIBILITY - DROP YOUR TRACKS - ";

const fileNameWithoutExt = (file) => file?.name?.replace(/\.[^/.]+$/, "");

export default function Home() {
  const [song1, setSong1] = useState(null);
  const [song2, setSong2] = useState(null);
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
      const previewFile = backendResult.preview_source === "song2" ? song2 : song1;

      setResult({
        ...backendResult,
        preview_url: previewFile ? URL.createObjectURL(previewFile) : null,
        preview_song_name: backendResult.preview_song_name || fileNameWithoutExt(previewFile),
      });
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
    setResult(null);
    setErrorMessage("");
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <div className="w-full bg-foreground text-background overflow-hidden py-2 border-b-2 border-foreground">
        <div className="flex animate-marquee whitespace-nowrap">
          {[...Array(4)].map((_, i) => (
            <span key={i} className="font-mono text-xs tracking-widest mr-0 px-0">
              {MARQUEE_TEXT}
            </span>
          ))}
        </div>
      </div>

      <div className="flex-1 flex flex-col md:flex-row">
        <div className="md:w-2/5 border-r-2 border-foreground flex flex-col justify-between p-8 md:p-12 min-h-[40vh]">
          <div>
            <div className="w-8 h-8 rounded-full bg-primary border-2 border-foreground mb-8" />
            <h1 className="font-display text-[clamp(5rem,12vw,9rem)] leading-none text-foreground">
              MASH
              <br />
              <span className="text-primary">LAB</span>
            </h1>
            <p className="font-mono text-xs text-muted-foreground mt-6 max-w-xs leading-relaxed">
              Drop two tracks. We&apos;ll tell you if they belong together and let you hear it.
            </p>
          </div>

          <div className="hidden md:grid grid-cols-4 gap-1 mt-auto pt-12">
            {[...Array(16)].map((_, i) => (
              <div
                key={i}
                className="h-2 rounded-sm"
                style={{
                  background:
                    i % 3 === 0
                      ? "hsl(var(--primary))"
                      : i % 5 === 0
                        ? "hsl(var(--foreground))"
                        : "hsl(var(--border))",
                  opacity: 0.4 + (i % 4) * 0.15,
                }}
              />
            ))}
          </div>
        </div>

        <div className="md:w-3/5 flex flex-col p-8 md:p-12 gap-8">
          {!result ? (
            <>
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-muted-foreground tracking-widest uppercase">
                  01 - Upload Tracks
                </span>
                <div className="flex-1 h-px bg-border" />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <SongUploader
                  label="TRACK A"
                  file={song1}
                  onFileChange={setSong1}
                  disabled={isAnalyzing}
                  index="A"
                />
                <SongUploader
                  label="TRACK B"
                  file={song2}
                  onFileChange={setSong2}
                  disabled={isAnalyzing}
                  index="B"
                />
              </div>

              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-border" />
                <span className="font-mono text-xs text-muted-foreground tracking-widest uppercase">
                  02 - Analyze
                </span>
                <div className="flex-1 h-px bg-border" />
              </div>

              <button
                onClick={handleAnalyze}
                disabled={!canAnalyze}
                className="brutal-btn w-full py-5 text-xl tracking-widest font-display uppercase flex items-center justify-center gap-3"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    ANALYZING...
                  </>
                ) : (
                  "RUN COMPATIBILITY CHECK ->"
                )}
              </button>

              {!canAnalyze && !isAnalyzing && (
                <p className="font-mono text-xs text-muted-foreground text-center -mt-4">
                  {!song1 && !song2
                    ? "Upload both tracks to continue"
                    : !song1
                      ? "Still need Track A"
                      : "Still need Track B"}
                </p>
              )}

              {errorMessage && (
                <p className="font-mono text-xs text-destructive text-center">{errorMessage}</p>
              )}
            </>
          ) : (
            <>
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-muted-foreground tracking-widest uppercase">
                  03 - Results
                </span>
                <div className="flex-1 h-px bg-border" />
              </div>

              <ResultCard result={result} song1Name={song1?.name} song2Name={song2?.name} />

              <button
                onClick={handleReset}
                className="font-mono text-xs text-muted-foreground hover:text-foreground transition-colors self-start border-b border-muted-foreground hover:border-foreground pb-0.5"
              >
                {"<- try different tracks"}
              </button>
            </>
          )}
        </div>
      </div>

      <div className="border-t-2 border-foreground px-8 py-3 flex justify-between items-center">
        <span className="font-mono text-xs text-muted-foreground">MASHLAB v1.0</span>
        <div className="flex gap-1">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="w-2 h-2 rounded-full bg-border"
              style={{ background: i < 2 ? "hsl(var(--primary))" : undefined }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
