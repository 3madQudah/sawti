import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { LANGUAGES, type Language, type Schemas } from "../api/client";
import { useQueue } from "../api/hooks";
import { CoverageBar } from "../components/CoverageBar";
import { LangChip } from "../components/LangChip";
import { LoadError, Loading } from "../components/Status";
import { useI18n } from "../i18n/I18nProvider";
import { fixed, shortId, waitingFor } from "../lib/format";

type Item = Schemas["ReviewQueueItem"];

/** Re-render once a minute so "waiting" stays current. */
function useNow(): Date {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 60_000);
    return () => clearInterval(timer);
  }, []);
  return now;
}

export function EscalationText({ item }: { item: Pick<Item, "escalation" | "escalation_reason"> }) {
  const { t } = useI18n();
  const escalation = item.escalation;
  if (!escalation) return <span dir="auto">{item.escalation_reason ?? "—"}</span>;
  if (escalation.kind === "analysis_error") return <>{t("reason.analysis_error")}</>;
  return <>{t("reason.low_grounding", { rejected: escalation.rejected_claims, total: escalation.total_claims })}</>;
}

export function QueuePage() {
  const { t } = useI18n();
  const now = useNow();
  const queue = useQueue();
  const [params, setParams] = useSearchParams();
  const raw = params.get("language");
  const filter: Language | null = LANGUAGES.includes(raw as Language) ? (raw as Language) : null;

  const items = useMemo(() => queue.data?.items ?? [], [queue.data]);
  const pending = queue.data?.pending;
  const rows = filter === null ? items : items.filter((item) => item.language === filter);

  const oldest = useMemo(() => {
    const out: Partial<Record<Language, string>> = {};
    for (const item of items) {
      const since = item.waiting_since;
      if (!since) continue;
      const current = out[item.language];
      if (current === undefined || since < current) out[item.language] = since;
    }
    return out;
  }, [items]);

  const thresholds = [...new Set(items.flatMap((i) => (i.escalation?.threshold == null ? [] : [i.escalation.threshold])))];

  const filters: { value: Language | null; label: string; count: number | undefined }[] = [
    { value: null, label: t("lang.all"), count: pending ? pending.ar + pending.en + pending.mixed : undefined },
    ...LANGUAGES.map((language) => ({ value: language, label: t(`lang.${language}`), count: pending?.[language] })),
  ];

  return (
    <main className="mx-auto flex w-full max-w-[1280px] flex-col gap-6 p-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex max-w-[760px] flex-col gap-1.5">
          <h1 className="m-0 text-[28px] font-semibold">{t("queue.title")}</h1>
          <p className="m-0 leading-relaxed text-muted">{t("queue.subtitle")}</p>
        </div>
        <div role="group" aria-label={t("queue.filter")} className="flex flex-wrap gap-1.5">
          {filters.map((f) => {
            const on = filter === f.value;
            return (
              <button
                key={f.value ?? "all"}
                type="button"
                aria-pressed={on}
                onClick={() => setParams(f.value === null ? {} : { language: f.value })}
                className={`min-h-10 rounded-lg border px-3.5 font-medium ${on ? "border-ink bg-ink text-white" : "border-[#D3D7D2] bg-white text-ink"}`}
              >
                {f.label} <span className="font-mono opacity-75">{f.count ?? "–"}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {LANGUAGES.map((language) => (
          <section
            key={language}
            aria-label={t(`lang.${language}`)}
            className="flex flex-col gap-2.5 rounded-[10px] border border-line bg-white px-5 py-[18px]"
          >
            <div className="flex items-center justify-between">
              <LangChip language={language} />
              <span className="text-[13px] text-muted">{t(`lang.${language}`)}</span>
            </div>
            <div className="flex items-baseline gap-2">
              <span data-testid={`pending-${language}`} className="text-[32px] font-semibold">
                {pending?.[language] ?? "–"}
              </span>
              <span className="text-muted">{t("queue.pending")}</span>
            </div>
            <div className="text-[13px] text-muted">
              {t("queue.oldest")}{" "}
              <span dir="ltr" className="font-mono text-ink">
                {waitingFor(oldest[language], now)}
              </span>
            </div>
          </section>
        ))}
      </div>

      {queue.isPending && <Loading />}
      {queue.isError && <LoadError error={queue.error} retry={() => void queue.refetch()} />}
      {queue.isSuccess && (
        <div className="overflow-x-auto rounded-[10px] border border-line bg-white">
          <table className="w-full min-w-[880px] border-collapse text-start">
            <thead>
              <tr className="text-xs tracking-[0.04em] text-muted">
                {(["call", "agent", "language", "coverage", "reason", "waiting"] as const).map((col) => (
                  <th key={col} className="border-b border-line px-5 py-3.5 text-start font-medium">
                    {t(`queue.col.${col}`)}
                  </th>
                ))}
                <th className="border-b border-line px-5 py-3.5">
                  <span className="sr-only">{t("queue.col.action")}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((item) => (
                <tr key={item.call_id} className="border-b border-line-soft last:border-b-0">
                  <td dir="ltr" className="px-5 py-4 text-start font-mono text-[13px]" title={item.call_id}>
                    {shortId(item.call_id)}
                  </td>
                  <td className="px-5 py-4">{item.agent_name}</td>
                  <td className="px-5 py-4">
                    <LangChip language={item.language} />
                  </td>
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-2.5">
                      <CoverageBar coverage={item.grounding_coverage} threshold={item.escalation?.threshold} />
                      <span className="font-mono text-[13px]">{fixed(item.grounding_coverage)}</span>
                    </div>
                  </td>
                  <td className="px-5 py-4 text-muted">
                    <EscalationText item={item} />
                  </td>
                  <td dir="ltr" className="px-5 py-4 text-start font-mono text-[13px]">
                    {waitingFor(item.waiting_since, now)}
                  </td>
                  <td className="px-5 py-2.5 text-end">
                    <Link
                      to={`/calls/${item.call_id}`}
                      className="inline-flex min-h-10 items-center rounded-lg bg-teal px-4 font-medium text-white no-underline hover:bg-teal-dark hover:text-white"
                    >
                      {t("queue.review")}
                    </Link>
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-5 py-10 text-center text-muted">
                    {filter === null ? t("queue.empty") : t("queue.emptyFiltered")}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      <p className="m-0 text-[13px] text-muted">
        {thresholds.length === 1 && <>{t("queue.threshold", { threshold: fixed(thresholds[0]) })} · </>}
        {thresholds.length > 1 && <>{t("queue.thresholdVaries")} · </>}
        {t("queue.footer")}
      </p>
    </main>
  );
}
