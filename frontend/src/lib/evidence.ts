import type { TranscriptTurn } from "../api/client";

export interface EvidenceMatch {
  /** Offset of the first occurrence in the redacted transcript. */
  start: number;
  end: number;
  /** 1-based number of the turn the first occurrence starts in (as the UI shows turns). */
  turn: number;
  /** All occurrences. */
  occurrences: number;
}

/**
 * Find `evidence` verbatim in the redacted transcript: an exact substring, the
 * same rule as the backend's `sawti.quotes.is_verbatim`. No trimming, no
 * normalisation, so what the reviewer submits is exactly what is in the text.
 *
 * Returns null when it is empty or absent, or does not start inside a turn.
 */
export function findVerbatim(
  transcript: string,
  turns: readonly TranscriptTurn[],
  evidence: string,
): EvidenceMatch | null {
  if (evidence.length === 0) return null;
  const start = transcript.indexOf(evidence);
  if (start === -1) return null;
  const turn = turns.find((t) => t.start_char <= start && start < t.end_char);
  if (turn === undefined) return null;
  let occurrences = 0;
  for (let i = start; i !== -1; i = transcript.indexOf(evidence, i + 1)) occurrences += 1;
  return { start, end: start + evidence.length, turn: turn.index + 1, occurrences };
}

/** The verdict note carrying a verified corrected evidence span (the API cannot edit a quote). */
export function evidenceNote(evidence: string, match: EvidenceMatch): string {
  return `Corrected evidence (verbatim, turn ${match.turn}, chars ${match.start}-${match.end}): «${evidence}»`;
}
