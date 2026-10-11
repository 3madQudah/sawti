import { useState } from "react";
import { Link } from "react-router-dom";

import type { ConflictAction, Schemas } from "../../api/client";
import { useResolveConflict } from "../../api/hooks";
import { LangChip } from "../../components/LangChip";
import { useI18n } from "../../i18n/I18nProvider";
import { textDirection } from "../../lib/direction";
import { fixed, shortId } from "../../lib/format";

type Conflict = Schemas["MemoryConflictView"];
type Rule = Schemas["MemoryRuleView"];

const ACTIONS: readonly ConflictAction[] = ["keep_existing", "keep_new", "discard", "not_a_conflict"];

const day = (iso: string) => iso.slice(0, 10);

export function ConflictsView({ conflicts, rules }: { conflicts: Conflict[]; rules: Rule[] }) {
  const { t } = useI18n();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const resolve = useResolveConflict();
  const [last, setLast] = useState<Schemas["ConflictResolveResponse"] | null>(null);
  const selected = conflicts.find((c) => c.id === selectedId) ?? conflicts[0] ?? null;

  return (
    <div className="flex flex-col gap-3">
      {last && (
        <p role="status" className="m-0 rounded-lg border border-teal/30 bg-teal-tint px-4 py-2.5 text-sm text-teal-dark">
          {t("memory.resolved", { action: t(`memory.action.${last.action}`) })}
          {last.consolidation_queued && <> {t("memory.consolidationQueued")}</>}
        </p>
      )}
      {conflicts.length === 0 ? (
        <p className="m-0 rounded-[10px] border border-line bg-white px-5 py-10 text-center text-muted">{t("memory.noConflicts")}</p>
      ) : (
        <div className="grid grid-cols-[minmax(260px,340px)_1fr] gap-4">
          <ul aria-label={t("memory.conflicts")} className="m-0 list-none self-start overflow-hidden rounded-[10px] border border-line bg-white p-0">
            {conflicts.map((c) => {
              const on = selected?.id === c.id;
              return (
                <li key={c.id}>
                  <button
                    type="button"
                    aria-current={on}
                    onClick={() => {
                      setSelectedId(c.id);
                      resolve.reset();
                    }}
                    className={`flex w-full flex-col gap-1.5 border-0 border-b border-s-[3px] border-b-line-soft px-4 py-3.5 text-start ${on ? "border-s-amber bg-amber-tint" : "border-s-transparent bg-white"}`}
                  >
                    <span className="flex items-center gap-2">
                      <span dir="ltr" className="font-mono text-xs text-muted">{shortId(c.id)}</span>
                      <LangChip language={c.source.language} />
                    </span>
                    <span dir={textDirection(c.candidate_text)} className="line-clamp-2 text-start text-sm">{c.candidate_text}</span>
                  </button>
                </li>
              );
            })}
          </ul>
          {selected && (
            <ConflictDetail
              key={selected.id}
              conflict={selected}
              rules={rules}
              pending={resolve.isPending}
              error={resolve.error}
              onResolve={(action) =>
                resolve.mutate(
                  { conflictId: selected.id, action },
                  {
                    onSuccess: (response) => {
                      setLast(response);
                      setSelectedId(null);
                    },
                  },
                )
              }
            />
          )}
        </div>
      )}
    </div>
  );
}

function ConflictDetail({
  conflict,
  rules,
  pending,
  error,
  onResolve,
}: {
  conflict: Conflict;
  rules: Rule[];
  pending: boolean;
  error: Error | null;
  onResolve: (action: ConflictAction) => void;
}) {
  const { t } = useI18n();
  const [action, setAction] = useState<ConflictAction>("keep_existing");
  const byId = new Map(rules.map((r) => [r.id, r]));

  return (
    <section aria-label={t("memory.conflictHeader", { id: shortId(conflict.id), date: day(conflict.detected_at) })} className="flex flex-col gap-4 rounded-[10px] border border-line bg-white p-5">
      <div className="font-mono text-xs tracking-[0.04em] text-muted">
        {t("memory.conflictHeader", { id: shortId(conflict.id), date: day(conflict.detected_at) })}
      </div>

      {conflict.conflicting.map((existing) => {
        const rule = byId.get(existing.rule_id);
        return (
          <div key={existing.rule_id} data-testid="existing-rule" className="flex flex-col gap-1.5 rounded-lg border border-line px-4 py-3">
            <div className="flex flex-wrap items-center gap-2 text-xs font-medium tracking-[0.04em] text-muted">
              <span>{t("memory.existing")}</span>
              <span dir="ltr" className="font-mono">{shortId(existing.rule_id)}</span>
              <span dir="ltr" className="font-mono">· {t("memory.similarity", { value: fixed(existing.similarity) })}</span>
              {!existing.active && <span className="rounded-full bg-rubric px-2 py-0.5">{t("memory.retired")}</span>}
            </div>
            <p dir={existing.text ? textDirection(existing.text) : undefined} className="m-0 text-start leading-relaxed">{existing.text ?? t("memory.textGone")}</p>
            <p dir={textDirection(existing.reasoning)} className="m-0 text-start text-[13px] text-muted">{existing.reasoning}</p>
            {rule && (
              <p className="m-0 text-xs text-muted">{t("memory.usage", { sources: rule.source_corrections, uses: rule.analyses_used })}</p>
            )}
          </div>
        );
      })}

      <div data-testid="proposed-rule" className="flex flex-col gap-1.5 rounded-lg border border-amber-edge bg-amber-tint px-4 py-3">
        <div className="text-xs font-medium tracking-[0.04em] text-amber-text">{t("memory.proposed")}</div>
        <p dir={textDirection(conflict.candidate_text)} className="m-0 text-start leading-relaxed">{conflict.candidate_text}</p>
        <p className="m-0 flex flex-wrap items-center gap-1.5 text-xs text-muted">
          {t("memory.source", { reviewer: conflict.source.reviewer_id, date: day(conflict.source.corrected_at) })}
          <Link dir="ltr" to={`/calls/${conflict.source.call_id}`} className="font-mono">{shortId(conflict.source.call_id)}</Link>
          <LangChip language={conflict.source.language} />
        </p>
      </div>

      <fieldset className="m-0 flex flex-col gap-2 border-0 p-0">
        <legend className="mb-2 font-medium">{t("memory.resolution")}</legend>
        <div className="grid grid-cols-2 gap-2">
          {ACTIONS.map((a) => (
            <label
              key={a}
              className={`flex min-h-14 cursor-pointer gap-2.5 rounded-lg border px-3 py-2.5 ${action === a ? "border-ink" : "border-field"}`}
            >
              <input type="radio" name="resolution" value={a} checked={action === a} onChange={() => setAction(a)} className="mt-1" />
              <span className="flex flex-col gap-0.5">
                <span className="font-medium">{t(`memory.action.${a}`)}</span>
                <span className="text-xs text-muted">{t(`memory.action.${a}.hint`)}</span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      {error && (
        <p role="alert" className="m-0 text-sm text-danger">{t("memory.resolveError", { message: error.message })}</p>
      )}
      <button
        type="button"
        disabled={pending}
        onClick={() => onResolve(action)}
        className="min-h-11 self-start rounded-lg bg-ink px-5 font-semibold text-white disabled:bg-muted"
      >
        {pending ? t("memory.resolving") : t("memory.resolve")}
      </button>
    </section>
  );
}
