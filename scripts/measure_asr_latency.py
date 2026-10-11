"""ASR latency and real-time factor per language, on Phase 3 audio, with the service's pipeline.

Phase 6.4 A1. Runs `sawti.asr.pipeline.transcribe_call`, the function the
native ASR worker runs (ffmpeg decode, mlx-whisper with the forced language,
pyannote diarization constrained to two speakers, role labelling), on the
first `--per-language` calls of each category in `data/audio/manifest.json`
(by call id: deterministic, not chosen by length).

Cold start is measured apart and not counted. By default: loading the
pyannote pipeline, then one warm-up transcription on a call outside the
measured set (Whisper's model loads inside its first call). With `--calls`:
both models are loaded explicitly once (pyannote, and Whisper through
mlx-whisper's own model cache), with no warm-up transcription.

`--cooldown-s` pauses between files, so each one starts from an idle machine
rather than after minutes of sustained load (the M1 is fanless-class under
this load and throttles). Every row records its `condition`. `--prior` merges
rows measured earlier, e.g. calls recorded before an interrupted run.

Writes per-call timings and per-language summaries to
`data/asr_latency/<date>_<host>.json`: no transcript text, only counts and
seconds. Real-time factor (RTF) = processing time / audio duration; below 1 is
faster than real time.

Usage:
    uv run python scripts/measure_asr_latency.py [--per-language 3]
    uv run python scripts/measure_asr_latency.py --calls call_0003_mixed call_0004_mixed \
        --cooldown-s 180 --prior data/asr_latency/earlier_rows.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sawti.asr.diarization import load_pipeline
from sawti.asr.pipeline import transcribe_call
from sawti.config import get_settings
from sawti.schemas import Language

MANIFEST = Path("data/audio/manifest.json")
OUT_DIR = Path("data/asr_latency")


def select_calls(
    manifest: dict[str, Any], per_language: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """The first `per_language` calls of each category by id, plus one more call for the warm-up."""
    entries = sorted(manifest["entries"], key=lambda e: e["call_id"])
    chosen = {
        lang.value: [e for e in entries if e["language"] == lang.value][:per_language] for lang in Language
    }
    used = {e["call_id"] for calls in chosen.values() for e in calls}
    warmup = next(e for e in entries if e["call_id"] not in used)
    return chosen, warmup


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-language aggregates over per-call rows.

    Rows from an earlier run may lack the per-stage split (`transcribe_s`,
    `diarize_s` = None); the stage RTFs pool only rows that have it, with
    their n stated.
    """
    audio = sum(r["audio_s"] for r in rows)
    total = sum(r["total_s"] for r in rows)
    staged = [r for r in rows if r.get("transcribe_s") is not None and r.get("diarize_s") is not None]
    staged_audio = sum(r["audio_s"] for r in staged)
    return {
        "n": len(rows),
        "audio_s_total": round(audio, 1),
        "processing_s_mean": round(statistics.mean(r["total_s"] for r in rows), 1),
        "processing_s_median": round(statistics.median(r["total_s"] for r in rows), 1),
        "processing_s_max": round(max(r["total_s"] for r in rows), 1),
        "rtf_pooled": round(total / audio, 3),
        "stage_split_n": len(staged),
        "rtf_whisper_pooled": round(sum(r["transcribe_s"] for r in staged) / staged_audio, 3)
        if staged
        else None,
        "rtf_diarization_pooled": round(sum(r["diarize_s"] for r in staged) / staged_audio, 3)
        if staged
        else None,
        "rtf_per_call": [r["rtf"] for r in rows],
        "conditions": sorted({r.get("condition", "unknown") for r in rows}),
    }


def _sysctl(name: str) -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware() -> dict[str, Any]:
    """What the numbers were measured on."""
    memsize = _sysctl("hw.memsize")
    return {
        "machine": platform.machine(),
        "cpu": _sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(memsize) / 2**30, 1) if memsize.isdigit() else memsize,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "whisper_backend": "mlx-whisper (Metal GPU)",
        "whisper_model": get_settings().whisper_model,
        "diarization": "pyannote/speaker-diarization-3.1, CPU (torch default device), num_speakers=2",
        "versions": {
            pkg: version(pkg) for pkg in ("mlx-whisper", "mlx", "pyannote.audio", "torch", "torchaudio")
        },
    }


def _measure(entry: dict[str, Any], pipe: Any) -> dict[str, Any]:
    result = transcribe_call(Path(entry["audio_path"]), category=entry["language"], diarization_pipeline=pipe)
    lines = result.raw_transcript.splitlines()
    return {
        "call_id": entry["call_id"],
        "language": entry["language"],
        "audio_s": round(result.audio_seconds, 2),
        "decode_s": round(result.timings.decode_s, 2),
        "transcribe_s": round(result.timings.transcribe_s, 2),
        "diarize_s": round(result.timings.diarize_s, 2),
        "total_s": round(result.timings.total_s, 2),
        "rtf": round(result.realtime_factor, 3),
        "turns": len(lines),
        "speakers": sorted({line.split(":", 1)[0] for line in lines}),
        "decoded_language": result.decoded_language,
    }


def preload_whisper() -> float:
    """Load the Whisper model into mlx-whisper's own cache (what `transcribe` reuses); return seconds."""
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    from sawti.asr.transcribe import _MLX_REPOS

    model = get_settings().whisper_model
    started = time.perf_counter()
    ModelHolder.get_model(_MLX_REPOS.get(model, model), mx.float16)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--per-language", type=int, default=3)
    parser.add_argument("--calls", nargs="+", default=None, help="measure only these call ids, in this order")
    parser.add_argument("--cooldown-s", type=float, default=0.0, help="idle pause before each file")
    parser.add_argument("--prior", type=Path, default=None, help="JSON list of rows measured earlier")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("measure_asr_latency")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    chosen, warmup = select_calls(manifest, args.per_language)

    started = time.perf_counter()
    pipe = load_pipeline()
    cold_start: dict[str, Any] = {"pyannote_load_s": round(time.perf_counter() - started, 1)}
    log.info("pyannote pipeline loaded in %.1fs", cold_start["pyannote_load_s"])
    if args.calls:
        by_id = {e["call_id"]: e for e in manifest["entries"]}
        todo = [by_id[call_id] for call_id in args.calls]
        cold_start["whisper_load_s"] = round(preload_whisper(), 1)
        log.info("whisper model loaded in %.1fs", cold_start["whisper_load_s"])
    else:
        todo = [e for entries in chosen.values() for e in entries]
        cold_start["warmup_call"] = _measure(warmup, pipe)
        log.info("warm-up (not counted) %s: %.1fs", warmup["call_id"], cold_start["warmup_call"]["total_s"])

    rows: list[dict[str, Any]] = json.loads(args.prior.read_text(encoding="utf-8")) if args.prior else []
    for index, entry in enumerate(todo):
        if args.cooldown_s > 0:
            log.info("cooldown %.0fs before %s", args.cooldown_s, entry["call_id"])
            time.sleep(args.cooldown_s)
        row = _measure(entry, pipe)
        row["condition"] = (
            f"after {args.cooldown_s:.0f}s cooldown" if args.cooldown_s > 0 else "warm, no cooldown"
        ) + ("; first file after model load" if index == 0 else "")
        rows.append(row)
        log.info(
            "%s (%s): %.1fs audio in %.1fs, RTF %.3f [%s]",
            row["call_id"],
            row["language"],
            row["audio_s"],
            row["total_s"],
            row["rtf"],
            row["condition"],
        )

    languages = [lang.value for lang in Language if any(r["language"] == lang.value for r in rows)]
    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": (
            "sawti.asr.pipeline.transcribe_call (the ASR worker's function), one file at a time, models "
            "loaded once; RTF = processing time / audio duration; first calls by id per language; "
            "synthetic TTS audio (Phase 3, ar-JO / en voices); per-row `condition`"
        ),
        "hardware": hardware(),
        "cold_start": cold_start,
        "per_language": {lang: summarize([r for r in rows if r["language"] == lang]) for lang in languages},
        "calls": rows,
    }
    out = args.out or OUT_DIR / f"{datetime.now(UTC).date().isoformat()}_{platform.node().split('.')[0]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["per_language"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
