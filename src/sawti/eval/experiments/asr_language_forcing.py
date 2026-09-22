"""Phase 3 experiment: forced Arabic decoding vs Whisper's own language detection.

Phase 3: ASR language-forcing validation.

`sawti.asr.transcribe` forces `language="ar"` rather than letting Whisper
detect, and `docs/08-ROADMAP.md` asserts that detection "degrades badly on
code-switched audio". This experiment exists so that claim is measured on this
corpus rather than assumed.

Each call is transcribed twice — once with the language forced, once with
`AUTO_DETECT` — and both hypotheses are scored against the same reference with
the same normalization. It is aimed at the `mixed` category, where Whisper
commits to a language from the opening 30 seconds and then decodes the rest of
a bilingual call under that guess, but it accepts any call ids so the `ar` and
`en` categories can be checked for a regression too.

Results are appended to `eval_results.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from sawti.asr.transcribe import AUTO_DETECT, transcribe
from sawti.data.generate_audio import DEFAULT_MANIFEST_PATH, AudioManifest
from sawti.eval.metrics import word_error_rate
from sawti.schemas import Language

logger = logging.getLogger(__name__)


@dataclass
class CallComparison:
    """One call transcribed both ways, with both WERs."""

    call_id: str
    language: Language
    forced_language: str
    detected_language: str
    forced_wer: float
    detected_wer: float
    forced_wer_raw: float
    detected_wer_raw: float
    forced_sec: float
    detected_sec: float


@dataclass
class ForcingReport:
    """Aggregate of a forced-vs-detected run, always per language category."""

    comparisons: list[CallComparison]

    def detected_language_counts(self) -> dict[Language, dict[str, int]]:
        """How often Whisper detected each language, per category.

        The failure this experiment is looking for: a `mixed` call detected as
        `en` is one whose Arabic half gets transliterated into Latin script.
        """
        counts: dict[Language, dict[str, int]] = {}
        for comparison in self.comparisons:
            bucket = counts.setdefault(comparison.language, {})
            bucket[comparison.detected_language] = bucket.get(comparison.detected_language, 0) + 1
        return counts


def compare_call(
    audio_path: Path, reference_text: str, language: Language, forced_language: str = "ar"
) -> CallComparison:
    """Transcribe one call forced and auto-detected, and score both.

    Args:
        audio_path: Path to the call audio.
        reference_text: The true transcript text for this call.
        language: The call's language category.
        forced_language: Language code to force in the forced arm.

    Returns:
        The two-way comparison for this call.
    """
    started = time.perf_counter()
    forced = transcribe(audio_path, language=forced_language)
    forced_sec = time.perf_counter() - started

    started = time.perf_counter()
    detected = transcribe(audio_path, language=AUTO_DETECT)
    detected_sec = time.perf_counter() - started

    return CallComparison(
        call_id=audio_path.stem,
        language=language,
        forced_language=forced_language,
        detected_language=detected.language,
        forced_wer=word_error_rate(reference_text, forced.text),
        detected_wer=word_error_rate(reference_text, detected.text),
        forced_wer_raw=word_error_rate(reference_text, forced.text, normalize=False),
        detected_wer_raw=word_error_rate(reference_text, detected.text, normalize=False),
        forced_sec=forced_sec,
        detected_sec=detected_sec,
    )


def compare_against_recorded_forced(
    audio_path: Path,
    reference_text: str,
    language: Language,
    *,
    forced_wer: float,
    forced_wer_raw: float,
    forced_language: str = "ar",
) -> CallComparison:
    """Run only the detection arm, pairing it with an already-recorded forced result.

    The corpus transcription pass in `sawti.eval.experiments.asr_propagation`
    already decodes every call with the language forced, and Whisper decodes
    greedily at temperature 0 — so re-running the forced arm here would repeat
    identical work for ~45 minutes. This reuses that result and transcribes
    only the auto-detect arm.

    Both arms must come from the same model and the same code path for the
    comparison to mean anything; the caller is responsible for that, which is
    why this is a separate function rather than a flag on `compare_call`.

    Args:
        audio_path: Path to the call audio.
        reference_text: The true transcript text for this call.
        language: The call's language category.
        forced_wer: Normalized WER already measured with the language forced.
        forced_wer_raw: Unnormalized WER from the same forced run.
        forced_language: The language code that forced run used.

    Returns:
        The two-way comparison for this call.
    """
    started = time.perf_counter()
    detected = transcribe(audio_path, language=AUTO_DETECT)
    detected_sec = time.perf_counter() - started

    return CallComparison(
        call_id=audio_path.stem,
        language=language,
        forced_language=forced_language,
        detected_language=detected.language,
        forced_wer=forced_wer,
        detected_wer=word_error_rate(reference_text, detected.text),
        forced_wer_raw=forced_wer_raw,
        detected_wer_raw=word_error_rate(reference_text, detected.text, normalize=False),
        forced_sec=0.0,
        detected_sec=detected_sec,
    )


def run_forcing_experiment(
    manifest: AudioManifest,
    *,
    categories: set[Language] | None = None,
    limit: int | None = None,
    forced_language: str = "ar",
) -> ForcingReport:
    """Run the forced-vs-detected comparison over the manifest.

    Args:
        manifest: The synthesized audio manifest.
        categories: Language categories to include. `None` means all.
        limit: Maximum calls per category.
        forced_language: Language code to force in the forced arm.

    Returns:
        A `ForcingReport` over every call compared.
    """
    per_category_count: dict[Language, int] = {}
    comparisons: list[CallComparison] = []

    for entry in manifest.entries:
        if categories is not None and entry.language not in categories:
            continue
        seen = per_category_count.get(entry.language, 0)
        if limit is not None and seen >= limit:
            continue
        per_category_count[entry.language] = seen + 1

        comparison = compare_call(
            Path(entry.audio_path), entry.reference_text, entry.language, forced_language
        )
        comparisons.append(comparison)
        logger.info(
            "%s (%s): forced WER %.3f [%s] vs detected WER %.3f [detected %s]",
            comparison.call_id, comparison.language.value,
            comparison.forced_wer, comparison.forced_language,
            comparison.detected_wer, comparison.detected_language,
        )

    return ForcingReport(comparisons=comparisons)


def summarize(report: ForcingReport) -> dict[str, object]:
    """Reduce a report to per-category WER under both modes.

    Args:
        report: The completed comparison report.

    Returns:
        A JSON-serializable summary, per language category. Never blended.
    """
    summary: dict[str, object] = {}
    for language in Language:
        rows = [c for c in report.comparisons if c.language is language]
        if not rows:
            continue
        summary[language.value] = {
            "calls": len(rows),
            "forced_wer": round(sum(c.forced_wer for c in rows) / len(rows), 4),
            "detected_wer": round(sum(c.detected_wer for c in rows) / len(rows), 4),
            "forced_wer_raw": round(sum(c.forced_wer_raw for c in rows) / len(rows), 4),
            "detected_wer_raw": round(sum(c.detected_wer_raw for c in rows) / len(rows), 4),
            "detected_languages": report.detected_language_counts().get(language, {}),
        }
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the language-forcing experiment."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--categories", nargs="*", default=["mixed"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--forced-language", default="ar")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = AudioManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    categories = {Language(name) for name in args.categories} if args.categories else None

    report = run_forcing_experiment(
        manifest, categories=categories, limit=args.limit, forced_language=args.forced_language
    )
    summary = summarize(report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
