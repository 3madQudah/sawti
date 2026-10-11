/**
 * Grounding coverage as a bar. Amber when below the run's own threshold. With a
 * threshold, a tick marks it. `inset-inline-start`, so it grows from the reading
 * start in both directions.
 */
export function CoverageBar({
  coverage,
  threshold,
  className = "h-1.5 w-24",
  showTick = false,
}: {
  coverage: number | null | undefined;
  threshold: number | null | undefined;
  className?: string;
  showTick?: boolean;
}) {
  const value = Math.max(0, Math.min(1, coverage ?? 0));
  const below = threshold !== null && threshold !== undefined && value < threshold;
  return (
    <div
      role="meter"
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={value}
      className={`relative rounded-full bg-line ${className}`}
    >
      <div
        className={`absolute inset-y-0 start-0 rounded-full ${below ? "bg-amber" : "bg-teal"}`}
        style={{ width: `${Math.round(value * 100)}%` }}
      />
      {showTick && threshold !== null && threshold !== undefined && (
        <div
          data-testid="threshold-tick"
          className="absolute -inset-y-1 w-0.5 bg-ink"
          style={{ insetInlineStart: `${Math.round(threshold * 100)}%` }}
        />
      )}
    </div>
  );
}
