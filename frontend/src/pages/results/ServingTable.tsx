import type { Language, Schemas } from "../../api/client";
import { useI18n } from "../../i18n/I18nProvider";
import { fixed } from "../../lib/format";

type Serving = Schemas["ServingResults"];

const COLUMNS = ["arm", "hardware", "accuracy", "schemaValid", "grounding", "latency", "throughput", "cost", "egress"] as const;

export function ServingTable({ serving, language }: { serving: Serving; language: Language }) {
  const { t } = useI18n();
  return (
    <div className="overflow-x-auto rounded-[10px] border border-line">
      <table className="w-full min-w-[1080px] border-collapse text-start text-sm">
        <thead>
          <tr className="text-xs tracking-[0.04em] text-muted">
            {COLUMNS.map((col) => (
              <th key={col} className="border-b border-line px-4 py-3 text-start font-medium">
                {t(`results.col.${col}`)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {serving.arms.map((arm) => {
            const c = arm[language];
            return (
              <tr key={arm.id} data-testid={`arm-${arm.id}`} className="border-b border-line-soft align-top last:border-b-0">
                <td lang="en" dir="ltr" className="px-4 py-3 text-start font-medium">{arm.label}</td>
                <td className="px-4 py-3">
                  <div>{t(`results.deployment.${arm.deployment}`)}</div>
                  <div lang="en" dir="ltr" className="text-start text-xs text-muted">{arm.hardware}</div>
                </td>
                <td dir="ltr" data-testid="accuracy" className="px-4 py-3 text-start font-mono">{fixed(c.accuracy, 3)}</td>
                <td dir="ltr" data-testid="schema-valid" className="px-4 py-3 text-start font-mono">
                  {t("results.cell.valid", { valid: c.schema_valid, n: c.n })}
                </td>
                <td dir="ltr" className="px-4 py-3 text-start font-mono">{fixed(c.grounding_precision, 3)}</td>
                <td className="px-4 py-3">
                  <div dir="ltr" data-testid="latency" className="text-start font-mono">
                    {t("results.cell.latency", { p50: fixed(c.latency_p50_s, 1), p95: fixed(c.latency_p95_s, 1) })}
                  </div>
                  <div dir="ltr" className="text-start text-xs text-muted">{t("results.cell.latencyN", { n: c.latency_n })}</div>
                </td>
                <td className="px-4 py-3">
                  <div data-testid="throughput" className="font-mono">
                    {t("results.cell.throughput", { rate: fixed(arm.throughput.calls_per_min, 3) })}
                  </div>
                  <div className="text-xs text-muted">{t("results.cell.inFlight", { concurrency: arm.throughput.concurrency })}</div>
                </td>
                <td className="px-4 py-3">
                  <div dir="ltr" data-testid="cost" className="text-start font-mono">
                    {c.cost_per_1000_usd == null ? "—" : `$${fixed(c.cost_per_1000_usd)}`}
                  </div>
                  {c.cost_per_1000_usd_sensitivity != null && (
                    <div className="text-xs text-muted">{t("results.cell.sensitivity", { cost: fixed(c.cost_per_1000_usd_sensitivity) })}</div>
                  )}
                </td>
                <td className="px-4 py-3">
                  <div className="font-mono">
                    {c.third_party_transcript_bytes_per_call == null
                      ? "—"
                      : t("results.cell.bytes", { bytes: Math.round(c.third_party_transcript_bytes_per_call).toLocaleString("en-US") })}
                  </div>
                  <div className="text-xs text-muted">{t(`results.egress.${arm.deployment}`)}</div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
