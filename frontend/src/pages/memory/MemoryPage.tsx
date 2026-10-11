import { useSearchParams } from "react-router-dom";

import { LANGUAGES } from "../../api/client";
import { useConflicts, useRules, useStats } from "../../api/hooks";
import { LangChip } from "../../components/LangChip";
import { LoadError, Loading } from "../../components/Status";
import { useI18n } from "../../i18n/I18nProvider";
import { ConflictsView } from "./ConflictsView";
import { RulesTable } from "./RulesTable";

type Tab = "conflicts" | "rules";

export function MemoryPage() {
  const { t } = useI18n();
  const [params, setParams] = useSearchParams();
  const tab: Tab = params.get("tab") === "rules" ? "rules" : "conflicts";
  const conflicts = useConflicts();
  const rules = useRules();
  const stats = useStats();

  const tabs: { id: Tab; label: string; count: number | undefined }[] = [
    { id: "conflicts", label: t("memory.tab.conflicts"), count: conflicts.data?.conflicts.length },
    { id: "rules", label: t("memory.tab.rules"), count: rules.data?.rules.length },
  ];

  return (
    <main className="mx-auto flex w-full max-w-[1280px] flex-col gap-6 p-8">
      <div className="flex max-w-[760px] flex-col gap-1.5">
        <h1 className="m-0 text-[28px] font-semibold">{t("memory.title")}</h1>
        <p className="m-0 leading-relaxed text-muted">{t("memory.subtitle")}</p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {LANGUAGES.map((language) => {
          const s = stats.data?.[language];
          const fp = s?.conflict_false_positive_rate;
          return (
            <section key={language} aria-label={t(`lang.${language}`)} className="flex flex-col gap-2 rounded-[10px] border border-line bg-white px-5 py-4">
              <div className="flex items-center justify-between">
                <LangChip language={language} />
                <span className="text-[13px] text-muted">{t(`lang.${language}`)}</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span data-testid={`rules-${language}`} className="text-[28px] font-semibold">{s?.active_rules ?? "–"}</span>
                <span className="text-muted">{t("memory.tile.rules")}</span>
              </div>
              <div className="flex flex-col gap-0.5 text-[13px] text-muted">
                <span data-testid={`conflicts-${language}`}>{t("memory.tile.conflicts", { count: s?.pending_conflicts ?? "–" })}</span>
                {fp && (
                  <span data-testid={`fp-${language}`}>
                    {fp.n === 0 ? t("memory.tile.noResolutions") : t("memory.tile.falsePositives", { numerator: fp.numerator, n: fp.n })}
                  </span>
                )}
              </div>
            </section>
          );
        })}
      </div>
      {stats.isError && <LoadError error={stats.error} retry={() => void stats.refetch()} />}
      {stats.data && (
        <p className="-mt-3 m-0 text-[13px] text-muted">{t("memory.tile.note", { distinct: stats.data.active_rules_distinct })}</p>
      )}

      <div role="tablist" aria-label={t("memory.tabs")} className="flex gap-1.5 border-b border-line">
        {tabs.map((item) => {
          const on = tab === item.id;
          return (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={on}
              onClick={() => setParams(item.id === "rules" ? { tab: "rules" } : {})}
              className={`-mb-px min-h-11 border-b-2 px-4 font-medium ${on ? "border-ink text-ink" : "border-transparent text-muted"}`}
            >
              {item.label} <span className="font-mono opacity-75">{item.count ?? "–"}</span>
            </button>
          );
        })}
      </div>

      {tab === "conflicts" && (
        <div role="tabpanel">
          {(conflicts.isPending || rules.isPending) && <Loading />}
          {conflicts.isError && <LoadError error={conflicts.error} retry={() => void conflicts.refetch()} />}
          {conflicts.isSuccess && rules.isSuccess && (
            <ConflictsView conflicts={conflicts.data.conflicts} rules={rules.data.rules} />
          )}
        </div>
      )}
      {tab === "rules" && (
        <div role="tabpanel">
          {rules.isPending && <Loading />}
          {rules.isError && <LoadError error={rules.error} retry={() => void rules.refetch()} />}
          {rules.isSuccess && <RulesTable rules={rules.data.rules} />}
        </div>
      )}
    </main>
  );
}
