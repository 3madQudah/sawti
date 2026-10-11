import { useState } from "react";

import { LANGUAGES, type Language, type Schemas } from "../../api/client";
import { useResults } from "../../api/hooks";
import { LoadError, Loading } from "../../components/Status";
import { useI18n } from "../../i18n/I18nProvider";
import { fixed } from "../../lib/format";
import { MemoryEffect } from "./MemoryEffect";
import { ServingTable } from "./ServingTable";
import { ServingTiles } from "./ServingTiles";

/**
 * The 6.3 measurements. Every number on this page comes from `GET /results`,
 * which serves the recorded files unchanged; nothing is computed here except
 * ratios and differences of those numbers, and no language is ever blended.
 */
export function ResultsPage() {
  const { t } = useI18n();
  const results = useResults();
  const [language, setLanguage] = useState<Language>("mixed");

  return (
    <main className="mx-auto flex w-full max-w-[1280px] flex-col gap-6 p-8">
      <div className="flex max-w-[760px] flex-col gap-1.5">
        <h1 className="m-0 text-[28px] font-semibold">{t("results.title")}</h1>
        <p className="m-0 leading-relaxed text-muted">{t("results.subtitle")}</p>
      </div>

      {results.isPending && <Loading />}
      {results.isError && <LoadError error={results.error} retry={() => void results.refetch()} />}
      {results.isSuccess && (
        <>
          <MemoryEffect memory={results.data.memory} />
          <section aria-labelledby="serving-title" className="flex flex-col gap-4 rounded-[10px] border border-line bg-white p-6">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 id="serving-title" className="m-0 text-lg font-semibold">
                {t("results.serving.title")}
              </h2>
              <div role="group" aria-label={t("queue.filter")} className="flex gap-1.5">
                {LANGUAGES.map((l) => (
                  <button
                    key={l}
                    type="button"
                    aria-pressed={language === l}
                    onClick={() => setLanguage(l)}
                    className={`min-h-9 rounded-lg border px-3 font-mono font-semibold ${language === l ? "border-ink bg-ink text-white" : "border-[#D3D7D2] bg-white text-ink"}`}
                  >
                    <span dir="ltr">{l}</span>
                  </button>
                ))}
              </div>
            </div>
            <ServingTiles serving={results.data.serving} language={language} />
            <p data-testid="serving-language" className="m-0 text-sm text-muted">
              {t("results.serving.language", { language: t(`lang.${language}`) })}
            </p>
            <ServingTable serving={results.data.serving} language={language} />
            <Notes serving={results.data.serving} memory={results.data.memory} />
          </section>
        </>
      )}
    </main>
  );
}

function Notes({ serving, memory }: { serving: Schemas["ServingResults"]; memory: Schemas["MemoryResults"] }) {
  const { t } = useI18n();
  const spend = fixed(serving.actual_spend_usd, 0);
  return (
    <div className="flex flex-col gap-2 text-[13px] leading-relaxed text-muted">
      {!serving.accuracy_comparable_across_providers && <p className="m-0">{t("results.note.accuracy")}</p>}
      <p className="m-0">{t("results.note.cost", { spend })}</p>
      <p className="m-0">{t("results.note.egress")}</p>
      <p className="m-0">
        {t("results.note.source", { serving: serving.source, memory: memory.source })}
      </p>
      <details className="rounded-lg border border-line-soft px-4 py-3">
        <summary className="cursor-pointer font-medium text-ink">{t("results.method")}</summary>
        {/* Recorded provenance, verbatim and in English: it is the measurement's own record, not UI copy. */}
        <dl lang="en" dir="ltr" className="m-0 mt-3 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1.5 text-start">
          {Object.entries({ ...serving.methods, ...serving.pricing }).map(([key, text]) => (
            <div key={key} className="contents">
              <dt className="font-mono text-xs text-ink">{key}</dt>
              <dd className="m-0">{text}</dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  );
}
