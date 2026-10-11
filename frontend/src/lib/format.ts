/** Numbers and durations. Digits stay Western in both UI languages, as in the mockups. */

export function fixed(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined ? "—" : value.toFixed(digits);
}

/** "1h 42m" / "48m" / "<1m", from an ISO timestamp to `now`. */
export function waitingFor(since: string | null | undefined, now: Date): string {
  if (!since) return "—";
  const minutes = Math.max(0, Math.floor((now.getTime() - new Date(since).getTime()) / 60_000));
  if (minutes < 1) return "<1m";
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return hours > 0 ? `${hours}h ${String(rest).padStart(2, "0")}m` : `${rest}m`;
}

/** First 8 characters of a UUID: enough to tell rules and calls apart on screen. */
export function shortId(id: string): string {
  return id.slice(0, 8);
}
