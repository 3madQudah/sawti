import { LANGUAGES, type Schemas } from "../../api/client";
import { LangChip } from "../../components/LangChip";
import { useI18n } from "../../i18n/I18nProvider";
import { fixed } from "../../lib/format";

type Memory = Schemas["MemoryResults"];
type Row = Schemas["MemoryLanguageResult"];

export const signed = (v: number, digits = 3) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;

/** Isolate a number as left-to-right inside translated text, so "+0.056" keeps its sign in front in Arabic. */
const iso = (s: string) => `\u2066${s}\u2069`;

const STEP = 0.02;

/** An axis covering every effect, Phase 4 ± noise and zero, on 0.02 ticks. */
function axis(rows: Row[]) {
  const values = rows.flatMap((r) => [0, r.delta, r.phase4_delta - r.noise, r.phase4_delta + r.noise]);
  const lo = Math.floor(Math.min(...values) / STEP) * STEP - STEP / 2;
  const hi = Math.ceil(Math.max(...values) / STEP) * STEP;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / STEP) * STEP; v <= hi + 1e-9; v += STEP) ticks.push(Math.round(v * 100) / 100);
  return { x: (v: number) => ((v - lo) / (hi - lo)) * 100, ticks };
}

const VERDICT_STYLE: Record<Row["verdict"], string> = {
  confirms: "bg-teal text-white",
  contradicts: "bg-rubric text-subtle",
  inconclusive: "bg-rubric text-subtle",
  incomplete: "bg-amber-tint text-amber-text",
};

export function MemoryEffect({ memory }: { memory: Memory }) {
  const { t } = useI18n();
  const rows = LANGUAGES.map((language) => ({ language, row: memory[language] }));
  const { x, ticks } = axis(rows.map((r) => r.row));
  const repeats = Math.max(...rows.map((r) => r.row.delta_per_repeat.length));

  return (
    <section aria-labelledby="memory-title" className="flex flex-col gap-4 rounded-[10px] border border-line bg-white p-6">
      <div className="flex flex-col gap-1">
        <h2 id="memory-title" className="m-0 text-lg font-semibold">
          {t("results.memory.title")}
        </h2>
        <p className="m-0 text-[13px] text-muted">{t("results.memory.meta", { calls: memory.calls, repeats })}</p>
      </div>

      <div className="flex flex-wrap gap-5 text-[13px] text-muted">
        <span className="inline-flex items-center gap-2"><span className="inline-block h-2.5 w-5 rounded-sm bg-teal" />{t("results.memory.effect")}</span>
        <span className="inline-flex items-center gap-2"><span className="inline-block h-3 w-5 rounded-sm bg-amber-chip" />{t("results.memory.noise")}</span>
        <span className="inline-flex items-center gap-2"><span className="inline-block h-3.5 w-[3px] rounded-sm bg-amber" />{t("results.memory.phase4")}</span>
      </div>

      <div className="flex flex-col">
        {rows.map(({ language, row }) => (
          <div key={language} data-testid={`memory-${language}`} className="grid grid-cols-[180px_1fr_270px] items-center gap-5 border-b border-line-soft py-4 last:border-b-0">
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-2">
                <LangChip language={language} />
                <span className="font-medium">{t(`lang.${language}`)}</span>
              </div>
              <span className="text-xs text-muted">
                {t("results.memory.n", { n: row.n_paired_per_repeat.join(" / ") })}
              </span>
            </div>
            {/* A number line reads left to right in both UI languages. */}
            <div dir="ltr" className="relative h-9" aria-hidden="true">
              <div className="absolute inset-y-0 border-s border-dashed border-field" style={{ left: `${x(0)}%` }} />
              <div className="absolute top-1 bottom-1 rounded-sm bg-amber-chip" style={{ left: `${x(row.phase4_delta - row.noise)}%`, width: `${x(row.phase4_delta + row.noise) - x(row.phase4_delta - row.noise)}%` }} />
              <div className="absolute top-[11px] bottom-[11px] rounded-sm bg-teal" style={{ left: `${x(Math.min(0, row.delta))}%`, width: `${Math.abs(x(row.delta) - x(0))}%` }} />
              <div className="absolute top-0.5 bottom-0.5 -ms-px w-[3px] rounded-sm bg-amber" style={{ left: `${x(row.phase4_delta)}%` }} />
            </div>
            <div className="flex flex-col items-start gap-1.5">
              <div className="flex items-center gap-2">
                <span data-testid={`delta-${language}`} dir="ltr" className="font-mono text-lg font-semibold text-teal-dark">
                  {signed(row.delta)}
                </span>
                <span data-testid={`verdict-${language}`} className={`whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-semibold ${VERDICT_STYLE[row.verdict]}`}>
                  {t(`results.verdict.${row.verdict}`)}
                </span>
              </div>
              <span data-testid={`noise-${language}`} className="font-mono text-xs text-muted">
                {t("results.memory.values", { noise: iso(fixed(row.noise, 3)), phase4: iso(signed(row.phase4_delta)) })}
              </span>
            </div>
            <ul className="col-span-3 -mt-2 m-0 flex list-none flex-wrap gap-x-6 p-0 text-xs text-muted">
              {row.delta_per_repeat.map((delta, i) => (
                <li key={i} className="font-mono">
                  {t("results.memory.repeat", {
                    index: i + 1,
                    off: iso(fixed(row.accuracy_off_per_repeat[i], 3)),
                    on: iso(fixed(row.accuracy_on_per_repeat[i], 3)),
                    delta: iso(signed(delta)),
                  })}
                </li>
              ))}
            </ul>
          </div>
        ))}
        <div className="grid grid-cols-[180px_1fr_270px] gap-5" aria-hidden="true">
          <span />
          <div dir="ltr" className="relative h-5 font-mono text-[11px] text-muted">
            {ticks.map((tick) => (
              <span key={tick} className="absolute -translate-x-1/2" style={{ left: `${x(tick)}%` }}>
                {tick === 0 ? "0" : tick.toFixed(2)}
              </span>
            ))}
          </div>
        </div>
      </div>

      <p className="m-0 text-[13px] leading-relaxed text-muted">{t("results.memory.rule")}</p>
      <dl className="m-0 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-[13px] text-muted">
        <dt className="font-medium text-ink">{t("results.provider")}</dt>
        <dd lang="en" dir="ltr" className="m-0 text-start">{memory.provider}</dd>
        <dt className="font-medium text-ink">{t("results.hardware")}</dt>
        <dd lang="en" dir="ltr" className="m-0 text-start">{memory.hardware}</dd>
        <dt className="font-medium text-ink">{t("results.method")}</dt>
        <dd lang="en" dir="ltr" className="m-0 text-start">{memory.method}</dd>
      </dl>
    </section>
  );
}
