import type { Schemas } from "../../api/client";
import { useI18n } from "../../i18n/I18nProvider";
import { shortId } from "../../lib/format";

export function RejectedClaims({ claims }: { claims: readonly Schemas["RejectedClaim"][] }) {
  const { t } = useI18n();
  return (
    <section className="flex flex-col gap-2.5 rounded-[10px] border border-line bg-white px-[18px] py-4">
      <h3 className="m-0 text-sm font-semibold">
        {t("review.rejectedTitle")} <span className="font-normal text-muted">· {claims.length}</span>
      </h3>
      <p className="m-0 text-[13px] text-muted">{t("review.rejectedHelp")}</p>
      {claims.map((claim, i) => (
        <div key={i} className="flex flex-col gap-0.5 border-t border-line-soft pt-2">
          <span dir="auto" className="text-start font-medium">
            {claim.description}
          </span>
          <span dir="auto" className="text-start font-arabic text-danger-quote line-through">
            “{claim.attempted_quote}”
          </span>
        </div>
      ))}
    </section>
  );
}

export function RulesUsed({ rules }: { rules: readonly Schemas["MemoryRuleUsed"][] }) {
  const { t } = useI18n();
  return (
    <section className="flex flex-col gap-2.5 rounded-[10px] border border-line bg-white px-[18px] py-4">
      <h3 className="m-0 text-sm font-semibold">
        {t("review.rulesTitle")} <span className="font-normal text-muted">· {rules.length}</span>
      </h3>
      {rules.length === 0 && <p className="m-0 text-[13px] text-muted">{t("review.noRules")}</p>}
      {rules.map((rule) => (
        <div key={rule.id} className="flex items-baseline gap-2.5">
          <span dir="ltr" title={rule.id} className="flex-none font-mono text-xs text-teal">
            {shortId(rule.id)}
          </span>
          <span dir="auto" className="text-start text-[13px] leading-normal">
            {rule.text}
          </span>
        </div>
      ))}
    </section>
  );
}
