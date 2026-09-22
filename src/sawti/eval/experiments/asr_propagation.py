"""Phase 3 experiment: how ASR error propagates into extraction.

Phase 3: the headline measurement for the audio phase.

WER on its own does not say whether the system got worse. This experiment
answers the question that matters: run the *same* Phase 2 grounded agent over
ASR-produced transcripts instead of the clean Phase 1 text, and compare the
resulting accuracy and grounding numbers against the clean-text run already
recorded in `eval_results.md`. The delta, per language category, is what the
audio front end actually costs.

Isolating the variable
----------------------
Speaker labels come from the audio manifest's *true* speaker timeline, not
from diarization. That is deliberate. The clean transcripts are speaker-
labelled, so feeding the agent unlabelled ASR text would confound two effects —
losing the labels and mis-recognizing the words — and the resulting delta would
not be attributable to either. With oracle speaker attribution, the only thing
that changes between the clean run and this one is the words themselves, so the
delta is ASR text error and nothing else.

That makes this number a **lower bound** on the true cost of an audio front
end. Real deployment adds diarization error on top, which needs the gated
pyannote pipeline to measure; see `docs/09-DECISIONS.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sawti.asr.diarization import DiarizedSegment, merge_transcript_with_speakers
from sawti.asr.transcribe import TranscriptSegment, transcribe
from sawti.data.generate_audio import DEFAULT_MANIFEST_PATH, AudioManifest, AudioManifestEntry
from sawti.eval.experiments.grounded_graph import GateStats, extract_via_graph
from sawti.eval.metrics import word_error_rate
from sawti.eval.runner import run_eval
from sawti.schemas import CallAnalysis, Language

logger = logging.getLogger(__name__)

DEFAULT_ASR_TRANSCRIPT_DIR = Path("data/asr_transcripts")
DEFAULT_GROUND_TRUTH_DIR = Path("data/ground_truth")
DEFAULT_OUTPUT_PATH = Path("eval_results.md")
DEFAULT_WER_REPORT_PATH = Path("data/asr_transcripts/wer_report.json")

#: Provider-error retries per call. `run_eval` applies `max_retries` only to
#: its built-in phase 1 extractor, so a custom extractor — like this one — gets
#: no retry at all. On a 150-call run against a free-tier provider that means a
#: transient 503 permanently drops a call from the comparison, shrinking the
#: denominator against the phase 2 numbers this experiment exists to match.
EXTRACTION_ATTEMPTS = 4
EXTRACTION_BACKOFF_SEC = 20.0
EXPERIMENT_NAME = "phase 3 grounded agent on ASR transcripts (oracle speakers)"


@dataclass
class CallWer:
    """WER for one call, normalized and raw."""

    call_id: str
    language: str
    wer: float
    wer_raw: float
    reference_words: int
    hypothesis_words: int
    audio_sec: float
    transcribe_sec: float


@dataclass
class TranscriptionReport:
    """Outcome of transcribing the whole corpus."""

    calls: list[CallWer] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)

    def wer_by_category(self, *, normalize: bool = True) -> dict[Language, float]:
        """Corpus WER per language, word-weighted.

        Args:
            normalize: Report the normalized rate rather than the raw one.

        Returns:
            A mapping from each `Language` present to its WER.
        """
        errors: dict[Language, float] = {}
        words: dict[Language, int] = {}
        for call in self.calls:
            language = Language(call.language)
            rate = call.wer if normalize else call.wer_raw
            errors[language] = errors.get(language, 0.0) + rate * call.reference_words
            words[language] = words.get(language, 0) + call.reference_words
        return {
            language: (errors[language] / count if count else 0.0)
            for language, count in words.items()
        }

    def realtime_factor(self) -> float:
        """Audio seconds processed per wall-clock second, across the corpus.

        Calls restored from a previous run carry `transcribe_sec == 0.0` — they
        were not transcribed in this process — and are excluded from both sides
        of the ratio rather than counted as infinitely fast.
        """
        timed = [call for call in self.calls if call.transcribe_sec > 0.0]
        spent = sum(call.transcribe_sec for call in timed)
        return sum(call.audio_sec for call in timed) / spent if spent else 0.0


def _oracle_speaker_segments(entry: AudioManifestEntry) -> list[DiarizedSegment]:
    """The manifest's true speaker timeline, shaped as diarizer output.

    Lets the real alignment code path (`merge_transcript_with_speakers`) run
    unchanged while holding speaker attribution perfect.
    """
    return [
        DiarizedSegment(
            speaker=segment.speaker, start_sec=segment.start_sec, end_sec=segment.end_sec
        )
        for segment in entry.segments
    ]


def _render_labelled_transcript(
    entry: AudioManifestEntry, asr_segments: list[TranscriptSegment]
) -> str:
    """Render ASR segments as a speaker-labelled transcript.

    Matches the `Agent: ...` / `Customer: ...` line format of the Phase 1
    corpus, so the agent sees the same shape of input in both runs.
    Consecutive segments from one speaker are merged into a single turn.

    Args:
        entry: The manifest entry supplying the true speaker timeline.
        asr_segments: `TranscriptSegment`s from `transcribe()`.

    Returns:
        The speaker-labelled transcript text.
    """
    merged = merge_transcript_with_speakers(asr_segments, _oracle_speaker_segments(entry))

    lines: list[str] = []
    for segment in merged:
        text = segment.text.strip()
        if not text:
            continue
        if lines and lines[-1].startswith(f"{segment.speaker}:"):
            lines[-1] = f"{lines[-1]} {text}"
        else:
            lines.append(f"{segment.speaker}: {text}")
    return "\n".join(lines) + "\n"


_SPEAKER_PREFIX = re.compile(r"^(?:Agent|Customer):\s*")


def _hypothesis_from_transcript(text: str) -> str:
    """Recover the plain ASR hypothesis from a written labelled transcript.

    Strips the `Agent:` / `Customer:` prefixes this module added, leaving the
    recognized words — which is all WER needs. Lets a resumed run re-score an
    existing transcript instead of re-transcribing the audio.
    """
    return " ".join(
        _SPEAKER_PREFIX.sub("", line).strip()
        for line in text.splitlines()
        if line.strip()
    ).strip()


def transcribe_corpus(
    manifest: AudioManifest,
    *,
    output_dir: Path = DEFAULT_ASR_TRANSCRIPT_DIR,
    limit: int | None = None,
    resume: bool = True,
) -> TranscriptionReport:
    """Transcribe every call in `manifest` and write labelled ASR transcripts.

    Transcribing the corpus takes hours, so a call whose transcript is already
    on disk is re-scored from that file rather than re-transcribed. WER needs
    only the reference and hypothesis text, both of which survive in the
    written transcript, so a resumed run produces identical WER figures —
    it just cannot report a transcription time for those calls.

    Args:
        manifest: The synthesized audio manifest.
        output_dir: Directory to write `<call_id>.txt` ASR transcripts into.
        limit: Transcribe only the first N entries.
        resume: Re-score existing transcripts instead of re-transcribing them.

    Returns:
        A `TranscriptionReport` with per-call WER and timing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = manifest.entries[:limit] if limit else manifest.entries
    report = TranscriptionReport()

    for index, entry in enumerate(entries, start=1):
        existing = output_dir / f"{entry.call_id}.txt"
        if resume and existing.is_file():
            hypothesis = _hypothesis_from_transcript(existing.read_text(encoding="utf-8"))
            reference = entry.reference_text
            report.calls.append(
                CallWer(
                    call_id=entry.call_id,
                    language=entry.language.value,
                    wer=round(word_error_rate(reference, hypothesis), 4),
                    wer_raw=round(word_error_rate(reference, hypothesis, normalize=False), 4),
                    reference_words=len(reference.split()),
                    hypothesis_words=len(hypothesis.split()),
                    audio_sec=entry.duration_sec,
                    transcribe_sec=0.0,
                )
            )
            logger.info("[%d/%d] %s: already transcribed, re-scored", index, len(entries), entry.call_id)
            continue

        try:
            started = time.perf_counter()
            transcript = transcribe(Path(entry.audio_path), category=entry.language)
            elapsed = time.perf_counter() - started
        except Exception as error:  # one bad call must not end the corpus run
            logger.warning("%s: transcription failed: %s", entry.call_id, error)
            report.failures.append((entry.call_id, str(error)))
            continue

        reference = entry.reference_text
        output_dir.joinpath(f"{entry.call_id}.txt").write_text(
            _render_labelled_transcript(entry, transcript.segments), encoding="utf-8"
        )

        call = CallWer(
            call_id=entry.call_id,
            language=entry.language.value,
            wer=round(word_error_rate(reference, transcript.text), 4),
            wer_raw=round(word_error_rate(reference, transcript.text, normalize=False), 4),
            reference_words=len(reference.split()),
            hypothesis_words=len(transcript.text.split()),
            audio_sec=entry.duration_sec,
            transcribe_sec=round(elapsed, 2),
        )
        report.calls.append(call)
        logger.info(
            "[%d/%d] %s (%s): WER %.3f (%.1fs audio in %.1fs)",
            index, len(entries), call.call_id, call.language, call.wer,
            call.audio_sec, call.transcribe_sec,
        )

    return report


def _merge_gate_stats(target: GateStats, source: GateStats) -> None:
    """Fold `source`'s tallies into `target`.

    Used so a retried call contributes its gate counts exactly once: each
    attempt tallies into a scratch `GateStats`, and only the attempt that
    actually succeeded is merged.
    """
    target.calls += source.calls
    target.proposed += source.proposed
    target.rejected += source.rejected
    target.escalated += source.escalated
    for language, count in source.proposed_by_language.items():
        target.proposed_by_language[language] = target.proposed_by_language.get(language, 0) + count
    for language, count in source.rejected_by_language.items():
        target.rejected_by_language[language] = target.rejected_by_language.get(language, 0) + count


def extract_with_retry(
    transcript_path: Path,
    *,
    stats: GateStats | None = None,
    attempts: int = EXTRACTION_ATTEMPTS,
    backoff_sec: float = EXTRACTION_BACKOFF_SEC,
) -> CallAnalysis:
    """Run the phase 2 graph over `transcript_path`, retrying provider errors.

    `extract_via_graph` records its gate tally *before* it raises, so each
    attempt tallies into a scratch `GateStats` and only the successful attempt
    is merged into `stats` — otherwise a retried call would be counted twice.

    Args:
        transcript_path: Path to a `call_<idx>_<lang>.txt` transcript.
        stats: Optional tally to fold the successful attempt into.
        attempts: Maximum attempts before giving up.
        backoff_sec: Base delay, doubled after each failed attempt.

    Returns:
        The agent's analysis as a validated `CallAnalysis`.

    Raises:
        Exception: The last error, if every attempt failed.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        scratch = GateStats()
        try:
            analysis = extract_via_graph(transcript_path, stats=scratch)
        except Exception as error:  # provider faults surface as many exception types
            last_error = error
            if attempt < attempts:
                delay = backoff_sec * (2 ** (attempt - 1))
                logger.warning(
                    "%s attempt %d/%d failed (%s); retrying in %.0fs",
                    transcript_path.stem, attempt, attempts, error, delay,
                )
                time.sleep(delay)
            continue

        if stats is not None:
            _merge_gate_stats(stats, scratch)
        return analysis

    assert last_error is not None
    raise last_error


def run_asr_propagation(
    *,
    asr_transcript_dir: Path = DEFAULT_ASR_TRANSCRIPT_DIR,
    ground_truth_path: Path = DEFAULT_GROUND_TRUTH_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    sleep_seconds: float = 4.0,
    stats: GateStats | None = None,
) -> dict[str, dict[Language, float]]:
    """Run the Phase 2 agent over the ASR transcripts and score it.

    Identical to `run_grounded_graph` except that `synthetic_dir` points at the
    ASR transcripts rather than the clean ones — so grounding is verified
    against the text the model actually saw, which is the ASR output.

    Args:
        asr_transcript_dir: Directory of ASR-produced `<call_id>.txt` files.
        ground_truth_path: Directory of reference-label records.
        output_path: Where to append the results table.
        sleep_seconds: Delay between LLM calls, for free-tier rate limits.
        stats: Optional grounding-gate tally.

    Returns:
        A mapping from metric name to its per-`Language` category scores.
    """
    return run_eval(
        ground_truth_path,
        output_path=output_path,
        synthetic_dir=asr_transcript_dir,
        sleep_seconds=sleep_seconds,
        extractor=lambda path: extract_with_retry(path, stats=stats),
        experiment_name=EXPERIMENT_NAME,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: transcribe the corpus, then optionally score it."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asr-dir", type=Path, default=DEFAULT_ASR_TRANSCRIPT_DIR)
    parser.add_argument("--wer-report", type=Path, default=DEFAULT_WER_REPORT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-resume", action="store_true", help="Re-transcribe calls that already have a transcript."
    )
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Stop after transcription; do not run the extraction eval.",
    )
    args = parser.parse_args(argv)

    manifest = AudioManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    report = transcribe_corpus(
        manifest, output_dir=args.asr_dir, limit=args.limit, resume=not args.no_resume
    )

    args.wer_report.parent.mkdir(parents=True, exist_ok=True)
    args.wer_report.write_text(
        json.dumps(
            {
                "wer_by_category": {
                    language.value: round(rate, 4)
                    for language, rate in report.wer_by_category().items()
                },
                "wer_raw_by_category": {
                    language.value: round(rate, 4)
                    for language, rate in report.wer_by_category(normalize=False).items()
                },
                "realtime_factor": round(report.realtime_factor(), 2),
                "failures": report.failures,
                "calls": [asdict(call) for call in report.calls],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("WER by category: %s", report.wer_by_category())
    logger.info("Wrote WER report -> %s", args.wer_report)

    if args.transcribe_only:
        return

    stats = GateStats()
    scores = run_asr_propagation(asr_transcript_dir=args.asr_dir, stats=stats)
    logger.info("Extraction scores on ASR text: %s", scores)


if __name__ == "__main__":
    main()
