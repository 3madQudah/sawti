## Phase 1 
1. Foundation: src/sawti/schemas.py

The first file everything else builds on. Contains:

Quote: a verbatim transcript excerpt — text + speaker + start_char/end_char. Has a validator that enforces end_char - start_char == len(text), so a Quote can never be fabricated with mismatched offsets.
Claim: base class for any assertion (commitment, compliance flag, rubric score). All subclasses inherit a mandatory evidence: Quote. If the evidence is blank, validation rejects the object outright — not "low confidence," just invalid.
CallAnalysis: the top-level output (summary + commitments + compliance_flags + rubric_scores + sentiment_trajectory + confidence + requires_human_review). Enforces one important invariant: requires_human_review must be True when confidence is below a configured threshold — enforced at the schema level, not left to a caller to remember.
Language enum: ar/en/mixed — the axis everything is split on.
2. Data generation: src/sawti/data/generate_calls.py

This is where the 150 calls were generated:

generate_call() takes a language and a seed; the seed deterministically picks a scenario (call reason from 10 options, customer sentiment from 5, target length from 3) — not to pin the LLM's own sampling (which can vary even at temperature=0), but to pin the prompt sent to it.
A system prompt enforces the transcript format: every line is either Agent: or Customer:.
A real bug that was found and fixed: early Arabic transcripts leaked Egyptian dialect words (فندم، إزاي...) or "Arabizi" (Arabic transliterated into Latin letters, e.g. "wallah"). This was fixed with a strongly worded prompt constraint (_ARABIC_SCRIPT_CONSTRAINT, _JORDANIAN_DIALECT_CONSTRAINT) plus a _find_dialect_violations() check that scans every generated transcript and retries (exponential backoff) if it finds a banned word or a "glitch word" mixing Arabic and Latin script in one token.
_distribute_counts() splits the 150 calls into exactly 40%/20%/40% (ar/en/mixed) using the largest-remainder method, so the counts always sum to exactly 150 rather than approximately.
generate_batch() loops over all of them and writes data/synthetic/call_<idx>_<lang>.txt.
3. Ground truth — the important story: src/sawti/data/ground_truth.py

There are two paths here, and the file itself documents why both exist:

The original (abandoned) path: scripts/generate_ground_truth_templates.py writes 150 blank templates, meant to be filled in by a human reviewer using scripts/find_quote.py (a CLI tool) to compute offsets instead of counting characters by hand. This tooling is still in the repo but was never actually used for real — manually reviewing 150 calls wasn't feasible in the time available.

The actual path: generate_reference_labels(transcript_path):

Reads the transcript, infers call_id/language from the filename (infer_call_id_and_language).
Sends a long, detailed system prompt (_REFERENCE_SYSTEM_PROMPT) to Gemini, instructing it to act as an expert QA analyst — but with the one rule that matters most: evidence must come back as quote text only, never offsets. The model is never trusted to compute start_char/end_char.
The response (DraftAnalysis — a "draft" version of every shape, with evidence_quote: str instead of a full Quote) is then processed by locate_quote(), which searches for that text verbatim in the real transcript via sawti/quotes.py (find_quote_matches), and computes the real offset and speaker from there.
If a quote isn't found verbatim (the model paraphrased, translated, or altered it), that raises QuoteNotVerbatimError → retry, telling the model exactly which quotes it got wrong and asking it to copy them exactly this time, with exponential backoff up to 4 attempts.
Once every quote resolves, a final CallAnalysis is assembled and validated against the schema.

One real technical issue found during development: the first version of DraftAnalysis gave list fields a default (default_factory=list), which made those fields optional in the generated JSON Schema — and Gemini would sometimes omit rubric_scores entirely as a result. Fix: every list field is now required, with no default, and the reason is documented in the code.

4. src/sawti/quotes.py — the shared helper

A small, focused module (find_quote_matches) used in 3 places: find_quote.py (the abandoned manual CLI), generate_reference_labels (locating quote offsets), and metrics.py (grounding checks). The one rule this file exists to enforce: offsets are always computed from the real transcript text, never from the model and never by hand.

5. The baseline: src/sawti/eval/plain_extraction.py

This is the most important methodological point in the whole project. run_plain_extraction() uses the same LLM but with a completely different, deliberately short prompt: "Extract call_id, language, summary, commitments, compliance_flags, sentiment_trajectory, and score these 7 criteria... include the quote" — and nothing else. No instructions about verbatim quotes, no examples, no rubric calibration guidance, and no retry on content — whatever the model returns is the final number, as-is. Quotes that don't actually appear in the transcript are not corrected — that's exactly what grounding_precision exists to measure.

6. src/sawti/eval/metrics.py — the three metrics
accuracy_by_category: mean of 4 components per call: (a) summary presence, (b) _commitment_agreement (half count-match, half rough Jaccard token-overlap on description text), (c) binary agreement on whether compliance flags exist at all (not a detailed match, since rule_id vocabularies are model-invented and not comparable), (d) sentiment-direction agreement (up/down/flat/unknown).
rubric_agreement_by_category: 1 - MAE (mean absolute error) across every rubric criterion both sides scored.
grounding_precision_by_category: the fraction of quotes that actually appear verbatim in the transcript — the only metric that needs no ground truth at all, since it checks the prediction directly against the raw transcript.

Every one of these returns dict[Language, float] — enforced by the return type, so a single blended number can never sneak in.

7. src/sawti/eval/runner.py + src/sawti/eval/experiments/baseline.py

run_eval(): loads the reference labels, loops over every call, runs run_plain_extraction, computes all three metrics, and appends a new dated section to eval_results.md — with the LLM-generated caveat printed above the table, unconditionally. run_baseline() just wires this to the default paths (data/ground_truth and eval_results.md).

8. src/sawti/llm/provider.py

A thin abstraction over whatever LLM provider is configured (Gemini right now). run_with_timeout() was added after a real 9-minute hang on a single request with no deadline — every call is now bounded to 120 seconds.

Final result (from eval_results.md)
Metric	ar	en	mixed
Accuracy	0.893	0.858	0.874
Rubric agreement	0.954	0.954	0.910
Grounding precision	0.965	0.961	0.971

With the mandatory caveat: these are agreement scores between two LLM-driven processes, not true accuracy — except grounding precision, the one independent number. Full details and the decision itself are in docs/09-DECISIONS.md.