"""
render_mashup.py  (v2 CLI)

End-to-end CLI: analyze → score → find best segments → render → export.

Usage:
  python render_mashup.py song_a.mp3 song_b.mp3 \
    --duration 45 \
    --wav-out mashup.wav \
    --mp3-out mashup.mp3 \
    --analysis-out report.json
"""

from __future__ import annotations
import argparse
import json

from mashup_engine import MashupEngine, MashupConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a mashup preview clip from two songs.")
    parser.add_argument("track_a", type=str, help="Path to track A")
    parser.add_argument("track_b", type=str, help="Path to track B")
    parser.add_argument("--duration", type=float, default=45.0, help="Clip duration in seconds (default: 45)")
    parser.add_argument("--start-a", type=float, default=None, help="Manual start time for track A (seconds)")
    parser.add_argument("--start-b", type=float, default=None, help="Manual start time for track B (seconds)")
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "full_blend", "vocals_a_inst_b", "inst_a_vocals_b", "acapella_over_beat"],
                        help="Mashup blend mode")
    parser.add_argument("--stems", action="store_true", help="Use Demucs stem separation (slower, better quality)")
    parser.add_argument("--wav-out", type=str, default="mashup.wav")
    parser.add_argument("--mp3-out", type=str, default=None)
    parser.add_argument("--analysis-out", type=str, default=None)
    args = parser.parse_args()

    config = MashupConfig(
        track_a=args.track_a,
        track_b=args.track_b,
        clip_duration=args.duration,
        start_a=args.start_a,
        start_b=args.start_b,
        mashup_mode=args.mode,
        use_stem_separation=args.stems,
        wav_out=args.wav_out,
        mp3_out=args.mp3_out,
    )

    print("Analyzing tracks and finding best segments...")
    result = MashupEngine().run(config)

    if not result.success:
        print(f"ERROR: {result.error}")
        return

    print(f"\n{result.summary}")
    print(f"\nTrack A clip start: {result.start_a:.1f}s (quality: {result.segment_score_a:.2f})")
    print(f"Track B clip start: {result.start_b:.1f}s (quality: {result.segment_score_b:.2f})")
    print(f"Blend mode: {result.mashup_mode_used}")
    print(f"\nSaved WAV: {result.wav_out}")
    if result.mp3_out:
        print(f"Saved MP3: {result.mp3_out}")

    if args.analysis_out:
        comp = result.compatibility
        report = {
            "track_a": result.track_a_features,
            "track_b": result.track_b_features,
            "compatibility": {
                "score": comp.compatibility_score,
                "grade": comp.grade,
                "summary": comp.summary,
                "tempo_score": comp.tempo_score,
                "key_score": comp.key_score,
                "energy_score": comp.energy_score,
                "loudness_score": comp.loudness_score,
                "timbre_score": comp.timbre_score,
                "stretch_factor_b": comp.stretch_factor_b,
                "pitch_shift_b": comp.pitch_shift_b,
                "gain_db_b": comp.gain_db_b,
                "mashup_type": comp.mashup_type,
            },
            "render": {
                "start_a": result.start_a,
                "start_b": result.start_b,
                "duration_s": config.clip_duration,
                "mode": result.mashup_mode_used,
                "wav_out": result.wav_out,
                "mp3_out": result.mp3_out,
            },
        }
        with open(args.analysis_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved analysis: {args.analysis_out}")


if __name__ == "__main__":
    main()
