"""Synthetic bilingual call transcript generation for development and eval.

Phase 1: synthetic data generation.

Every model call goes through `sawti.llm.provider.get_llm_provider()` — this
module never imports a concrete LLM SDK client directly, so swapping
`SAWTI_LLM_PROVIDER` (e.g. anthropic <-> gemini) never requires touching
generation code.

Reproducibility: a provider may ignore `temperature` or sample
non-deterministically even at `temperature=0`, so `seed` does not attempt to
pin the LLM's own sampling. Instead `seed` deterministically selects the
call's *scenario* — reason, customer sentiment, and target length — via
`random.Random(seed)`, and that scenario is baked into the prompt. The same
seed therefore always produces the same prompt sent to the LLM; that is the
reproducibility guarantee this module makes.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import string
import time
from pathlib import Path

from sawti.llm.provider import get_llm_provider
from sawti.schemas import Language

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE_MIX: dict[Language, float] = {Language.AR: 0.4, Language.EN: 0.2, Language.MIXED: 0.4}

_CALL_REASONS = [
    "a billing dispute over an unexpected charge",
    "a service outage affecting internet or mobile data",
    "a complaint about a rude previous agent",
    "a refund request for a cancelled order",
    "a request to upgrade or change a subscription plan",
    "a delayed delivery or shipment",
    "a request to close an account",
    "a technical issue setting up a new device",
    "a dispute about a contract renewal",
    "a request for a payment extension",
]

_SENTIMENTS = [
    "calm and polite throughout",
    "frustrated and impatient",
    "confused and needing things repeated",
    "angry and escalating",
    "annoyed at first but satisfied by the end of the call",
]

_LENGTHS = [
    ("short, around 6-10 turns total", 8),
    ("medium, around 12-18 turns total", 15),
    ("long, around 20-28 turns total", 24),
]

# Shared, strongly worded constraints — dialect and quality problems found in a
# manual review of the generated dataset (Egyptian-dialect leakage, and Arabic
# written as Latin-letter "Arabizi" instead of Arabic script) recur often enough
# that they need to be stated explicitly and forcefully in every prompt that can
# produce Arabic content, not left implicit in "Jordanian/Gulf-neutral dialect".
_ARABIC_SCRIPT_CONSTRAINT = (
    "CRITICAL: every single Arabic word MUST be written in Arabic script (the "
    "Arabic Unicode alphabet), NEVER transliterated into Latin letters, under any "
    "circumstance. Writing Arabic words in Latin letters — e.g. 'wallah', 'shoo', "
    "'akeed', 'kif', 'ma'alesh', 'sidi' instead of والله، شو، أكيد، كيف، معلش، سيدي "
    "— is a serious error and strictly forbidden. It is likewise forbidden to mix "
    "Arabic and Latin letters inside a single word (e.g. a broken glitch-word like "
    "\"l'flوس\"). Every word must be entirely one script or the other."
)

_JORDANIAN_DIALECT_CONSTRAINT = (
    "CRITICAL: the Arabic must be Jordanian/Levantine dialect, NOT Egyptian "
    "Arabic. Never use Egyptian-dialect words or phrases, including: فندم، إزاي، "
    "خالص، معلش، عايز، فين، كده، النهاردة. Prefer the Jordanian/Levantine "
    "equivalent instead where one fits naturally — e.g. أستاذ / سيدي instead of "
    "فندم، كيف instead of إزاي، بدي instead of عايز، وين instead of فين، هيك "
    "instead of كده، اليوم instead of النهاردة."
)

_LANGUAGE_INSTRUCTIONS: dict[Language, str] = {
    Language.AR: (
        "Write the ENTIRE call fully in Arabic (Jordanian/Gulf-neutral dialect is "
        "fine). Do not use English words except for unavoidable brand/product names. "
        f"{_ARABIC_SCRIPT_CONSTRAINT} {_JORDANIAN_DIALECT_CONSTRAINT}"
    ),
    Language.EN: (
        "Write the ENTIRE call fully in English. This call has no Arabic content, "
        "so no Arabic script or dialect is expected — but if any Arabic word "
        "appears at all (e.g. in a name), it must still be written in Arabic "
        "script, never transliterated into Latin letters."
    ),
    Language.MIXED: (
        "Write the call in natural Arabic-English code-switching, the way a real "
        "regional (Jordan/Gulf) contact center agent and customer actually speak: "
        "switch between Arabic and English mid-sentence, not by alternating whole "
        "turns and not by concatenating two separate versions of the call. Mix "
        "naturally and a little inconsistently, the way real bilingual speakers do. "
        f"{_ARABIC_SCRIPT_CONSTRAINT} {_JORDANIAN_DIALECT_CONSTRAINT}"
    ),
}

_SYSTEM_PROMPT = (
    "You are generating synthetic training data: one realistic contact center call "
    "transcript for a telecom/utility company. Format every line as either "
    "'Agent: <utterance>' or 'Customer: <utterance>', one turn per line, with no "
    "other text, headers, or markdown."
)


def generate_call(language: Language, *, seed: int | None = None) -> str:
    """Generate a single synthetic call transcript.

    Args:
        language: Target language category for the generated call.
        seed: Optional RNG seed controlling the call scenario (reason,
            sentiment, length) for reproducibility. See module docstring.

    Returns:
        The generated transcript text, formatted with "Agent:" / "Customer:"
        speaker turns.
    """
    rng = random.Random(seed)
    reason = rng.choice(_CALL_REASONS)
    sentiment = rng.choice(_SENTIMENTS)
    length_desc, _target_turns = rng.choice(_LENGTHS)

    prompt = (
        f"Generate one realistic contact center call transcript about {reason}. "
        f"The customer's tone should be {sentiment}. "
        f"The call should be {length_desc}. "
        f"{_LANGUAGE_INSTRUCTIONS[language]}"
    )

    provider = get_llm_provider()
    transcript = asyncio.run(provider.complete(prompt, system=_SYSTEM_PROMPT, temperature=0.9))
    return transcript.strip()


# Egyptian-dialect markers that must never appear — see _JORDANIAN_DIALECT_CONSTRAINT.
_EGYPTIAN_MARKERS = ["فندم", "إزاي", "خالص", "معلش", "عايز", "فين", "كده", "النهاردة"]

_ARABIC_CHAR_RE = re.compile(r"[؀-ۿ]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_ARABIZI_APOSTROPHE_RE = re.compile(r"[A-Za-z]+'[A-Za-z]+")

# Real English contractions look identical to the "Latin-letters'Latin-letters"
# Arabizi pattern (e.g. "ba'dar", "l'order"); whitelist common ones so they
# aren't misflagged in the English portions of a code-switched call.
_ENGLISH_CONTRACTIONS = {
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "can't", "couldn't", "won't", "wouldn't", "shouldn't", "mustn't",
    "i'm", "i've", "i'd", "i'll", "you're", "you've", "you'd", "you'll",
    "we're", "we've", "we'd", "we'll", "they're", "they've", "they'd", "they'll",
    "he's", "she's", "it's", "that's", "there's", "here's", "what's", "who's",
    "let's", "ma'am", "y'all", "o'clock",
}

_PUNCTUATION_TO_STRIP = string.punctuation.replace("'", "") + "\u060c\u061b\u061f\u201c\u201d\u2018\u2019"


def _find_dialect_violations(text: str, language: Language) -> list[str]:
    """Return human-readable dialect/script problems found in `text`.

    An empty list means the transcript is clean. Checks:
      * Banned Egyptian-dialect marker words/phrases (any language).
      * For `language` in {AR, MIXED}: Arabizi — Arabic transliterated into
        Latin letters, detected via the apostrophe-adjacency heuristic
        described in `_ARABIC_SCRIPT_CONSTRAINT` (e.g. "kif ba'dar", "l'order").
      * A single whitespace-delimited token containing both an Arabic script
        character and a Latin letter — a broken mixed-script glitch-word
        (e.g. "l'flوس").
    """
    violations: list[str] = []

    for marker in _EGYPTIAN_MARKERS:
        if marker in text:
            violations.append(f"banned Egyptian-dialect word/phrase found: {marker!r}")

    for word in text.split():
        core = word.strip(_PUNCTUATION_TO_STRIP)
        if not core:
            continue

        if _ARABIC_CHAR_RE.search(core) and _LATIN_CHAR_RE.search(core):
            violations.append(f"mixed-script glitch word (Arabic+Latin in one token): {word!r}")

        if language in (Language.AR, Language.MIXED) and core.lower() not in _ENGLISH_CONTRACTIONS:
            if _ARABIZI_APOSTROPHE_RE.search(core) and not _ARABIC_CHAR_RE.search(core):
                violations.append(f"Arabizi (transliterated Arabic) detected in word: {word!r}")

    return violations


def _distribute_counts(n: int, language_mix: dict[Language, float]) -> dict[Language, int]:
    """Split `n` into per-language counts matching `language_mix` proportions exactly.

    Uses the largest-remainder method so per-language counts always sum to
    exactly `n`, even when `n * proportion` is not an integer.
    """
    raw = {lang: n * frac for lang, frac in language_mix.items()}
    counts = {lang: int(value) for lang, value in raw.items()}
    remainder = n - sum(counts.values())
    neediest_first = sorted(raw, key=lambda lang: raw[lang] - counts[lang], reverse=True)
    for lang in neediest_first[:remainder]:
        counts[lang] += 1
    return counts


def _generate_with_retry(language: Language, *, seed: int, max_retries: int, initial_delay: float) -> str:
    """Call `generate_call`, retrying with exponential backoff on provider errors
    or on a dialect/script violation (see `_find_dialect_violations`).

    Free-tier LLM APIs (e.g. Gemini) rate-limit aggressively; a transient 429
    or connection error should not abort an entire batch run. A transcript
    that leaks Egyptian dialect or Arabizi is treated the same way — as a
    retryable bad result, never returned silently.
    """
    delay = initial_delay
    for attempt in range(max_retries + 1):
        is_last_attempt = attempt == max_retries
        try:
            transcript = generate_call(language, seed=seed)
        except Exception:
            if is_last_attempt:
                raise
            logger.warning(
                "generate_call failed (attempt %d/%d) for seed=%d; retrying in %.1fs",
                attempt + 1,
                max_retries,
                seed,
                delay,
            )
            time.sleep(delay)
            delay *= 2
            continue

        violations = _find_dialect_violations(transcript, language)
        if not violations:
            return transcript
        if is_last_attempt:
            raise ValueError(
                f"generate_call for seed={seed} ({language.value}) kept producing dialect "
                f"violations after {max_retries} retries: {violations}"
            )
        logger.warning(
            "generate_call for seed=%d (%s) produced dialect violations (attempt %d/%d): %s; "
            "retrying in %.1fs",
            seed,
            language.value,
            attempt + 1,
            max_retries,
            violations,
            delay,
        )
        time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")  # pragma: no cover


def generate_batch(
    n: int,
    *,
    output_dir: Path,
    language_mix: dict[Language, float] | None = None,
    sleep_seconds: float = 4.0,
    max_retries: int = 5,
) -> list[Path]:
    """Generate a batch of synthetic call transcripts and write them to `output_dir`.

    Args:
        n: Number of calls to generate.
        output_dir: Directory to write generated transcripts into (created if missing).
        language_mix: Proportion of each language category; defaults to
            `DEFAULT_LANGUAGE_MIX` (40% ar / 20% en / 40% mixed).
        sleep_seconds: Delay between successive LLM calls, to respect
            free-tier rate limits. Not applied after the final call.
        max_retries: Retries (with exponential backoff) per call on provider errors.

    Returns:
        Paths to the written transcript files, one per generated call.
    """
    language_mix = language_mix or DEFAULT_LANGUAGE_MIX
    counts = _distribute_counts(n, language_mix)

    sequence: list[Language] = []
    for lang, count in counts.items():
        sequence.extend([lang] * count)
    random.Random(0).shuffle(sequence)  # avoid clumping all of one language together

    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for idx, language in enumerate(sequence):
        transcript = _generate_with_retry(
            language, seed=idx, max_retries=max_retries, initial_delay=sleep_seconds
        )
        path = output_dir / f"call_{idx:04d}_{language.value}.txt"
        path.write_text(transcript, encoding="utf-8")
        paths.append(path)
        if idx < len(sequence) - 1:
            time.sleep(sleep_seconds)
    return paths
