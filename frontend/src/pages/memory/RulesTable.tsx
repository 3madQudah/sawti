import type { Schemas } from "../../api/client";
import { LangChip } from "../../components/LangChip";
import { useI18n } from "../../i18n/I18nProvider";
import { textDirection } from "../../lib/direction";
import { shortId } from "../../lib/format";

export function RulesTable({ rules }: { rules: Schemas["MemoryRuleView"][] }) {
  const { t } = useI18n();
  return (
    <div className="flex flex-col gap-3">
      <div className="overflow-x-auto rounded-[10px] border border-line bg-white">
        <table className="w-full min-w-[880px] border-collapse text-start">
          <thead>
            <tr className="text-xs tracking-[0.04em] text-muted">
              {(["rule", "text", "languages", "sources", "used"] as const).map((col) => (
                <th key={col} className="border-b border-line px-5 py-3.5 text-start font-medium">
                  {t(`memory.col.${col}`)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rules.map((rule) => (
              <tr key={rule.id} className="border-b border-line-soft align-top last:border-b-0">
                <td dir="ltr" className="px-5 py-3.5 text-start font-mono text-[13px]" title={rule.id}>{shortId(rule.id)}</td>
                <td dir={textDirection(rule.text)} className="px-5 py-3.5 text-start leading-relaxed">{rule.text}</td>
                <td className="px-5 py-3.5">
                  <div className="flex flex-wrap gap-1">
                    {rule.languages.map((language) => <LangChip key={language} language={language} />)}
                  </div>
                </td>
                <td className="px-5 py-3.5 font-mono text-[13px]">{rule.source_corrections}</td>
                <td className="px-5 py-3.5 font-mono text-[13px]">{rule.analyses_used}</td>
              </tr>
            ))}
            {rules.length === 0 && (
              <tr>
                <td colSpan={5} className="px-5 py-10 text-center text-muted">{t("memory.rulesEmpty")}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="m-0 text-[13px] text-muted">{t("memory.rulesFooter", { count: rules.length })}</p>
    </div>
  );
}
