"""Tests for `sawti.eval.metrics`.

Phase 1: mirrors `src/sawti/eval/metrics.py`. Constructed `CallAnalysis`
fixtures with hand-computed expected scores — every metric must stay
per-language and must never collapse into a single blended number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sawti.data.ground_truth import RUBRIC_CRITERIA
from sawti.eval.metrics import (
    accuracy_by_category,
    grounding_precision_by_category,
    normalize_for_wer,
    rubric_agreement_by_category,
    speaker_attribution_accuracy,
    speaker_attribution_accuracy_by_category,
    unsupported_claim_rate_by_category,
    wer_by_category,
    word_error_rate,
)
from sawti.schemas import (
    CallAnalysis,
    Commitment,
    ComplianceFlag,
    Language,
    Quote,
    RubricScore,
    SentimentPoint,
    SentimentTrajectory,
    Severity,
)

TRANSCRIPT = (
    "Agent: Hello, you have reached support, my name is Rami.\n"
    "Customer: I was charged twice this month and I want a refund.\n"
    "Agent: I will process the refund today and send you a confirmation.\n"
    "Customer: Thank you, that helps.\n"
)

GROUNDED_QUOTE = "I will process the refund today"
UNGROUNDED_QUOTE = "I'll sort that out for you immediately"


def _quote(text: str = GROUNDED_QUOTE, speaker: str = "Agent") -> Quote:
    """A structurally valid Quote; offsets are not what the metrics check."""
    return Quote(text=text, speaker=speaker, start_char=0, end_char=len(text))


def _rubric(scores: dict[str, float] | None = None, quote_text: str = GROUNDED_QUOTE):
    values = scores or dict.fromkeys(RUBRIC_CRITERIA, 0.8)
    return [
        RubricScore(
            evidence=_quote(quote_text),
            criterion=criterion,
            score=score,
            justification="because",
        )
        for criterion, score in values.items()
    ]


def _analysis(
    call_id: str = "call_0000_ar",
    language: Language = Language.AR,
    *,
    summary: str = "Customer was double charged; agent promised a refund.",
    commitment_descriptions: tuple[str, ...] = ("process the refund today",),
    has_compliance_flag: bool = False,
    rubric: dict[str, float] | None = None,
    sentiment: tuple[float, ...] = (-0.6, 0.5),
    quote_text: str = GROUNDED_QUOTE,
) -> CallAnalysis:
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary=summary,
        commitments=[
            Commitment(evidence=_quote(quote_text), promised_by="Agent", description=description)
            for description in commitment_descriptions
        ],
        compliance_flags=(
            [
                ComplianceFlag(
                    evidence=_quote(quote_text),
                    rule_id="no_identity_verification",
                    severity=Severity.MEDIUM,
                    description="did not verify identity",
                )
            ]
            if has_compliance_flag
            else []
        ),
        rubric_scores=_rubric(rubric, quote_text),
        sentiment_trajectory=SentimentTrajectory(
            points=[
                SentimentPoint(
                    quote=_quote(quote_text, "Customer"),
                    speaker="Customer",
                    score=score,
                    timestamp_sec=float(index * 8),
                )
                for index, score in enumerate(sentiment)
            ]
        ),
        confidence=0.9,
        requires_human_review=False,
    )


@pytest.fixture
def transcript_dir(tmp_path: Path) -> Path:
    """A transcript directory holding one transcript per call_id used here."""
    for call_id in ("call_0000_ar", "call_0001_en", "call_0002_mixed"):
        (tmp_path / f"{call_id}.txt").write_text(TRANSCRIPT, encoding="utf-8")
    return tmp_path


# --- accuracy ------------------------------------------------------------


def test_accuracy_is_one_when_prediction_matches_reference():
    """Identical prediction and reference score a perfect 1.0 in their category."""
    reference = [_analysis()]

    assert accuracy_by_category([_analysis()], reference) == {Language.AR: 1.0}


def test_accuracy_groups_by_language_and_never_blends():
    """Each language gets its own score; a perfect en does not lift a broken ar."""
    reference = [
        _analysis("call_0000_ar", Language.AR),
        _analysis("call_0001_en", Language.EN),
    ]
    predictions = [
        # ar: wrong on commitments (none found), compliance, and sentiment direction
        _analysis(
            "call_0000_ar",
            Language.AR,
            commitment_descriptions=(),
            has_compliance_flag=True,
            sentiment=(0.5, -0.6),
        ),
        _analysis("call_0001_en", Language.EN),
    ]

    scores = accuracy_by_category(predictions, reference)

    assert set(scores) == {Language.AR, Language.EN}
    assert scores[Language.EN] == 1.0
    # ar: summary 1.0, commitments 0.0, compliance 0.0, sentiment 0.0 -> mean 0.25
    assert scores[Language.AR] == pytest.approx(0.25)


def test_accuracy_rewards_agreeing_that_a_call_has_no_commitments():
    """Both sides finding no commitments is correct, not an empty-vs-empty miss."""
    reference = [_analysis(commitment_descriptions=())]
    predictions = [_analysis(commitment_descriptions=())]

    assert accuracy_by_category(predictions, reference) == {Language.AR: 1.0}


def test_accuracy_scores_fuzzy_commitment_descriptions_partially():
    """A commitment described in different words scores between 0 and 1, not 0."""
    reference = [_analysis(commitment_descriptions=("process the refund today",))]
    predictions = [_analysis(commitment_descriptions=("refund the customer today",))]

    score = accuracy_by_category(predictions, reference)[Language.AR]

    # summary 1.0 + compliance 1.0 + sentiment 1.0 + partial commitment, over 4
    assert 0.75 < score < 1.0


def test_accuracy_ignores_calls_missing_from_either_side():
    """A prediction with no matching reference (or vice versa) is not scored."""
    reference = [_analysis("call_0000_ar", Language.AR)]
    predictions = [_analysis("call_0009_en", Language.EN)]

    assert accuracy_by_category(predictions, reference) == {}


# --- rubric agreement ----------------------------------------------------


def test_rubric_agreement_is_one_for_identical_scores():
    """Identical rubric scores give agreement 1.0 (MAE of 0)."""
    reference = [_analysis()]

    assert rubric_agreement_by_category([_analysis()], reference) == {Language.AR: 1.0}


def test_rubric_agreement_is_one_minus_mean_absolute_error():
    """Agreement is exactly 1 - MAE over the criteria scored on both sides."""
    reference = [_analysis(rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.8))]
    # every criterion off by 0.2 -> MAE 0.2 -> agreement 0.8
    predictions = [_analysis(rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.6))]

    scores = rubric_agreement_by_category(predictions, reference)

    assert scores[Language.AR] == pytest.approx(0.8)


def test_rubric_agreement_scores_zero_when_no_criteria_overlap():
    """A prediction that scored nothing comparable is a failure, not a skip."""
    reference = [_analysis()]
    predictions = [_analysis(rubric={"some_other_criterion": 0.8})]

    assert rubric_agreement_by_category(predictions, reference) == {Language.AR: 0.0}


def test_rubric_agreement_reports_each_language_separately():
    """Rubric agreement is per category, like every other metric here."""
    reference = [
        _analysis("call_0000_ar", Language.AR, rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.8)),
        _analysis("call_0002_mixed", Language.MIXED, rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.8)),
    ]
    predictions = [
        _analysis("call_0000_ar", Language.AR, rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.8)),
        _analysis("call_0002_mixed", Language.MIXED, rubric=dict.fromkeys(RUBRIC_CRITERIA, 0.3)),
    ]

    scores = rubric_agreement_by_category(predictions, reference)

    assert scores[Language.AR] == pytest.approx(1.0)
    assert scores[Language.MIXED] == pytest.approx(0.5)


# --- grounding precision -------------------------------------------------


def test_grounding_precision_is_one_when_every_quote_is_verbatim(transcript_dir):
    """Every claim quoting the transcript exactly scores a precision of 1.0."""
    predictions = [_analysis(quote_text=GROUNDED_QUOTE)]

    scores = grounding_precision_by_category(predictions, transcript_dir=transcript_dir)

    assert scores == {Language.AR: 1.0}


def test_grounding_precision_is_zero_for_invented_quotes(transcript_dir):
    """A quote that is not in the transcript is not grounded, whatever its offsets say."""
    predictions = [_analysis(quote_text=UNGROUNDED_QUOTE)]

    scores = grounding_precision_by_category(predictions, transcript_dir=transcript_dir)

    assert scores == {Language.AR: 0.0}


def test_grounding_precision_pools_claims_within_a_language(transcript_dir):
    """Precision is micro-averaged over claims, so claim-heavy calls weigh more."""
    grounded = _analysis("call_0000_ar", Language.AR, quote_text=GROUNDED_QUOTE)
    ungrounded = _analysis("call_0002_mixed", Language.MIXED, quote_text=UNGROUNDED_QUOTE)

    scores = grounding_precision_by_category([grounded, ungrounded], transcript_dir=transcript_dir)

    assert scores[Language.AR] == 1.0
    assert scores[Language.MIXED] == 0.0


def test_grounding_precision_needs_no_ground_truth(transcript_dir):
    """The metric takes predictions only — it is checked against the transcript."""
    scores = grounding_precision_by_category(
        [_analysis(quote_text=GROUNDED_QUOTE)], transcript_dir=transcript_dir
    )

    assert scores == {Language.AR: 1.0}


def test_grounding_precision_skips_calls_whose_transcript_is_missing(tmp_path):
    """A missing transcript contributes no claims, rather than scoring as perfect."""
    predictions = [_analysis("call_0000_ar", Language.AR)]

    assert grounding_precision_by_category(predictions, transcript_dir=tmp_path) == {}


def test_grounding_precision_counts_partially_grounded_calls(transcript_dir):
    """A call mixing real and invented quotes lands strictly between 0 and 1."""
    mixed = _analysis("call_0000_ar", Language.AR, quote_text=GROUNDED_QUOTE)
    mixed.compliance_flags = [
        ComplianceFlag(
            evidence=_quote(UNGROUNDED_QUOTE),
            rule_id="made_up",
            severity=Severity.LOW,
            description="invented",
        )
    ]

    score = grounding_precision_by_category([mixed], transcript_dir=transcript_dir)[Language.AR]

    assert 0.0 < score < 1.0


# --- unsupported claim rate ----------------------------------------------
#
# This metric is position-exact, unlike grounding precision: a quote must sit at
# the offsets it claims, not merely occur somewhere. The helpers above build
# quotes at start_char=0, so these tests construct their own.


def _located(text: str) -> Quote:
    """A quote whose offsets are computed from TRANSCRIPT — genuinely grounded."""
    start = TRANSCRIPT.index(text)
    return Quote(text=text, speaker="Agent", start_char=start, end_char=start + len(text))


def _misplaced(text: str) -> Quote:
    """A quote whose text is real but whose offsets point at the wrong span.

    Not unsupported: the grounding rule tests the text, and the graph re-anchors
    offsets it computes itself.
    """
    return Quote(text=text, speaker="Agent", start_char=0, end_char=len(text))


def _invented(text: str = UNGROUNDED_QUOTE) -> Quote:
    """A quote whose text appears nowhere in the transcript — genuinely unsupported."""
    return Quote(text=text, speaker="Agent", start_char=0, end_char=len(text))


def _with_commitments(call_id: str, language: Language, quotes: list[Quote]) -> CallAnalysis:
    """A CallAnalysis carrying one commitment per supplied quote and nothing else."""
    return CallAnalysis(
        call_id=call_id,
        language=language,
        summary="Summary.",
        commitments=[
            Commitment(evidence=quote, promised_by="Agent", description="refund") for quote in quotes
        ],
        confidence=1.0,
        requires_human_review=False,
    )


def test_unsupported_claim_rate_is_zero_when_every_claim_is_located(transcript_dir):
    """A pipeline that emits only verified claims scores 0.0 — lower is better here."""
    predictions = [_with_commitments("call_0000_ar", Language.AR, [_located(GROUNDED_QUOTE)])]

    rates = unsupported_claim_rate_by_category(predictions, transcript_dir=transcript_dir)

    assert rates[Language.AR] == pytest.approx(0.0)


def test_unsupported_claim_rate_is_one_when_every_claim_is_invented(transcript_dir):
    """Failure path: a paraphrased or invented quote is unsupported, all of it."""
    predictions = [_with_commitments("call_0000_ar", Language.AR, [_invented()])]

    rates = unsupported_claim_rate_by_category(predictions, transcript_dir=transcript_dir)

    assert rates[Language.AR] == pytest.approx(1.0)


def test_unsupported_claim_rate_forgives_wrong_offsets_on_real_text(transcript_dir):
    """Wrong offsets are not an unsupported claim — the metric follows the gate's rule.

    Measured on real model output, quote text verified ~100% of the time while
    start_char was right ~38% of the time. Counting offsets here would report a
    hallucination rate that is really an arithmetic rate.
    """
    predictions = [_with_commitments("call_0000_ar", Language.AR, [_misplaced(GROUNDED_QUOTE)])]

    rates = unsupported_claim_rate_by_category(predictions, transcript_dir=transcript_dir)

    assert rates[Language.AR] == pytest.approx(0.0)


def test_unsupported_claim_rate_complements_grounding_precision_on_claims(transcript_dir):
    """On claim-only input the two metrics are complements, because they share a rule."""
    predictions = [
        _with_commitments("call_0000_ar", Language.AR, [_located(GROUNDED_QUOTE), _invented()])
    ]

    precision = grounding_precision_by_category(predictions, transcript_dir=transcript_dir)
    rates = unsupported_claim_rate_by_category(predictions, transcript_dir=transcript_dir)

    assert precision[Language.AR] == pytest.approx(0.5)
    assert rates[Language.AR] == pytest.approx(1.0 - precision[Language.AR])


def test_unsupported_claim_rate_pools_claims_within_a_language(transcript_dir):
    """Half the claims unsupported means exactly 0.5, pooled across the language's calls."""
    predictions = [
        _with_commitments("call_0000_ar", Language.AR, [_located(GROUNDED_QUOTE)]),
        _with_commitments("call_0002_mixed", Language.MIXED, [_invented()]),
        _with_commitments("call_0001_en", Language.EN, [_located(GROUNDED_QUOTE), _invented()]),
    ]

    rates = unsupported_claim_rate_by_category(predictions, transcript_dir=transcript_dir)

    assert rates[Language.AR] == pytest.approx(0.0)
    assert rates[Language.MIXED] == pytest.approx(1.0)
    assert rates[Language.EN] == pytest.approx(0.5)


def test_unsupported_claim_rate_reports_languages_separately(transcript_dir):
    """Never a blended number: each language gets its own rate."""
    predictions = [
        _with_commitments("call_0000_ar", Language.AR, [_invented()]),
        _with_commitments("call_0001_en", Language.EN, [_located(GROUNDED_QUOTE)]),
    ]

    rates = unsupported_claim_rate_by_category(predictions, transcript_dir=transcript_dir)

    assert set(rates) == {Language.AR, Language.EN}
    assert rates[Language.AR] != rates[Language.EN]


def test_unsupported_claim_rate_skips_calls_whose_transcript_is_missing(tmp_path):
    """Failure path: a missing transcript must not look like a perfect score."""
    predictions = [_with_commitments("call_0000_ar", Language.AR, [_located(GROUNDED_QUOTE)])]

    rates = unsupported_claim_rate_by_category(predictions, transcript_dir=tmp_path)

    assert rates == {}


def test_unsupported_claim_rate_is_zero_for_a_call_with_no_claims(transcript_dir):
    """A call that claimed nothing has nothing unsupported — vacuously 0.0."""
    predictions = [_with_commitments("call_0000_ar", Language.AR, [])]

    rates = unsupported_claim_rate_by_category(predictions, transcript_dir=transcript_dir)

    assert rates == {}


# ---------------------------------------------------------------------------
# Phase 3: ASR metrics
# ---------------------------------------------------------------------------


class TestNormalizeForWer:
    """The orthographic normalization applied before Arabic WER scoring."""

    def test_strips_diacritics(self) -> None:
        """Harakat carry no lexical difference and are dropped."""
        assert normalize_for_wer("أَهْلاً بِكَ") == normalize_for_wer("أهلا بك")

    def test_folds_alef_variants(self) -> None:
        """أ إ آ ٱ all fold to bare alef."""
        assert normalize_for_wer("أحمد إبراهيم آدم") == "احمد ابراهيم ادم"

    def test_folds_ta_marbuta_and_alef_maqsura(self) -> None:
        """ة folds to ه and ى folds to ي."""
        assert normalize_for_wer("خدمة على") == "خدمه علي"

    def test_maps_arabic_indic_digits_to_ascii(self) -> None:
        """Digit script should not count as a recognition error."""
        assert normalize_for_wer("٠٧٩٥٥٤٣٢١٠") == "0795543210"

    def test_strips_punctuation_and_collapses_whitespace(self) -> None:
        """Punctuation, Arabic and Latin, is removed."""
        assert normalize_for_wer("مرحبا،  كيف   حالك؟") == "مرحبا كيف حالك"

    def test_lowercases_latin_for_code_switched_text(self) -> None:
        """The Latin half of a mixed call is case-folded too."""
        assert normalize_for_wer("Hello YA akhi") == "hello ya akhi"

    def test_orthographic_variants_collapse_to_the_same_string(self) -> None:
        """The whole point: two spellings of one utterance become identical."""
        assert normalize_for_wer("أهلاً بِكَ في خدمة العملاء،") == normalize_for_wer(
            "اهلا بك في خدمه العملاء"
        )


class TestWordErrorRate:
    """Single-pair WER."""

    def test_identical_text_scores_zero(self) -> None:
        """A perfect transcription has no errors."""
        assert word_error_rate("مرحبا كيف حالك", "مرحبا كيف حالك") == 0.0

    def test_one_substitution_in_three_words(self) -> None:
        """One wrong word out of three is a third."""
        assert word_error_rate("one two three", "one four three") == pytest.approx(1 / 3)

    def test_deletion_counts_as_an_error(self) -> None:
        """A dropped word is an error."""
        assert word_error_rate("one two three", "one three") == pytest.approx(1 / 3)

    def test_insertion_counts_as_an_error(self) -> None:
        """An invented word is an error."""
        assert word_error_rate("one two", "one two three") == pytest.approx(1 / 2)

    def test_normalization_is_applied_by_default(self) -> None:
        """Orthographic-only differences score 0.0 normalized."""
        assert word_error_rate("أهلاً بِكَ في خدمة", "اهلا بك في خدمه") == 0.0

    def test_raw_mode_charges_for_orthography(self) -> None:
        """Without normalization the same pair looks badly wrong."""
        raw = word_error_rate("أهلاً بِكَ في خدمة", "اهلا بك في خدمه", normalize=False)

        assert raw > 0.5

    def test_hallucination_may_exceed_one(self) -> None:
        """A runaway hypothesis scores worse than total loss, not clipped at 1.0."""
        assert word_error_rate("one", "one two three four five") > 1.0

    def test_empty_reference_against_empty_hypothesis_is_zero(self) -> None:
        """Nothing expected and nothing produced is not an error."""
        assert word_error_rate("", "") == 0.0

    def test_empty_reference_against_speech_is_total_error(self) -> None:
        """Nothing expected but something produced is total error."""
        assert word_error_rate("", "unexpected words") == 1.0


class TestWerByCategory:
    """Per-language WER aggregation."""

    def test_returns_a_rate_per_language_present(self) -> None:
        """Each category present in the input gets its own number."""
        result = wer_by_category(
            [
                (Language.AR, "مرحبا كيف حالك", "مرحبا كيف حالك"),
                (Language.EN, "one two three", "one four three"),
            ]
        )

        assert set(result) == {Language.AR, Language.EN}
        assert result[Language.AR] == 0.0
        assert result[Language.EN] == pytest.approx(1 / 3)

    def test_never_blends_languages_into_one_number(self) -> None:
        """Mixed must stay separable from ar and en — the project's standing rule."""
        result = wer_by_category(
            [
                (Language.AR, "a b c d", "a b c d"),
                (Language.MIXED, "a b c d", "w x y z"),
            ]
        )

        assert result[Language.AR] == 0.0
        assert result[Language.MIXED] == 1.0

    def test_aggregates_by_total_words_not_mean_of_call_rates(self) -> None:
        """A one-word call must not weigh as much as a long one.

        Per-call mean would give (1.0 + 0.0)/2 = 0.5; word-weighted gives
        1 error over 11 reference words.
        """
        result = wer_by_category(
            [
                (Language.EN, "x", "y"),
                (Language.EN, "a b c d e f g h i j", "a b c d e f g h i j"),
            ]
        )

        assert result[Language.EN] == pytest.approx(1 / 11)

    def test_empty_input_returns_empty_mapping(self) -> None:
        """No data means no categories, not a zero for every language."""
        assert wer_by_category([]) == {}


class TestSpeakerAttributionAccuracy:
    """The diarization sanity check."""

    def test_identical_timelines_score_one(self) -> None:
        """A perfect diarization attributes every speech frame correctly."""
        reference = [(0.0, 2.0, "Agent"), (2.5, 4.5, "Customer")]

        assert speaker_attribution_accuracy(reference, reference) == 1.0

    def test_fully_swapped_labels_score_zero(self) -> None:
        """Consistently inverted roles are wrong everywhere, not right everywhere."""
        reference = [(0.0, 2.0, "Agent"), (2.5, 4.5, "Customer")]
        hypothesis = [(0.0, 2.0, "Customer"), (2.5, 4.5, "Agent")]

        assert speaker_attribution_accuracy(reference, hypothesis) == 0.0

    def test_half_correct_scores_half(self) -> None:
        """Two equal-length turns, one right, is 0.5."""
        reference = [(0.0, 2.0, "Agent"), (2.5, 4.5, "Customer")]
        hypothesis = [(0.0, 2.0, "Agent"), (2.5, 4.5, "Agent")]

        assert speaker_attribution_accuracy(reference, hypothesis) == pytest.approx(0.5)

    def test_silence_between_turns_is_excluded(self) -> None:
        """Gaps must not inflate the score.

        The reference has a 10s silence after a 1s turn. A hypothesis that is
        correct on the speech still scores 1.0 — if silence counted, a
        hypothesis silent everywhere would score ~0.9 for saying nothing.
        """
        reference = [(0.0, 1.0, "Agent"), (11.0, 12.0, "Customer")]
        hypothesis = [(0.0, 1.0, "Agent"), (11.0, 12.0, "Customer")]

        assert speaker_attribution_accuracy(reference, hypothesis) == 1.0

    def test_empty_hypothesis_scores_zero(self) -> None:
        """Detecting no speakers at all is total failure, not vacuous success."""
        reference = [(0.0, 2.0, "Agent")]

        assert speaker_attribution_accuracy(reference, []) == 0.0

    def test_empty_reference_scores_zero(self) -> None:
        """No reference speech means there is nothing to have got right."""
        assert speaker_attribution_accuracy([], [(0.0, 2.0, "Agent")]) == 0.0


class TestSpeakerAttributionAccuracyByCategory:
    """Per-language aggregation of the diarization sanity check."""

    def test_returns_a_score_per_language_present(self) -> None:
        """Each category present gets its own number."""
        perfect = [(0.0, 2.0, "Agent")]
        swapped = [(0.0, 2.0, "Customer")]

        result = speaker_attribution_accuracy_by_category(
            [
                (Language.AR, perfect, perfect),
                (Language.MIXED, perfect, swapped),
            ]
        )

        assert result[Language.AR] == 1.0
        assert result[Language.MIXED] == 0.0

    def test_never_blends_languages(self) -> None:
        """mixed stays separable from ar — the project's standing rule."""
        perfect = [(0.0, 2.0, "Agent")]

        result = speaker_attribution_accuracy_by_category(
            [(Language.AR, perfect, perfect), (Language.EN, perfect, perfect)]
        )

        assert set(result) == {Language.AR, Language.EN}

    def test_empty_input_returns_empty_mapping(self) -> None:
        """No data means no categories."""
        assert speaker_attribution_accuracy_by_category([]) == {}


class TestApostropheNormalization:
    """Contractions must stay one token on both sides of the comparison."""

    def test_word_internal_apostrophe_is_deleted_not_spaced(self) -> None:
        """`let's` becomes `lets`, one token, not `let s`."""
        assert normalize_for_wer("Let's see") == "lets see"

    def test_contraction_against_apostrophe_free_asr_scores_zero(self) -> None:
        """The bug this rule fixes: ASR dropping the apostrophe is not 2 errors."""
        assert word_error_rate("Let's see", "lets see") == 0.0

    def test_typographic_apostrophe_is_handled_too(self) -> None:
        """Whisper emits U+2019, the reference may use U+0027."""
        assert normalize_for_wer("it’s fine") == normalize_for_wer("it's fine")

    def test_surrounding_quote_marks_are_still_stripped(self) -> None:
        """Only word-internal apostrophes are deleted; quotes remain punctuation."""
        assert normalize_for_wer("he said 'hello' loudly") == "he said hello loudly"

    def test_arabic_text_is_unaffected(self) -> None:
        """The rule targets Latin contractions and must not touch Arabic."""
        assert normalize_for_wer("أهلاً بك، كيف حالك؟") == "اهلا بك كيف حالك"
