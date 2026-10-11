import type { Language, Schemas } from "../../api/client";
import { useI18n } from "../../i18n/I18nProvider";
import { fixed } from "../../lib/format";

type Serving = Schemas["ServingResults"];
type Arm = Schemas["ServingArmResult"];

/** The arms the headline tiles compare, by their id in `comparison.json`. A missing arm drops its tile. */
const ARM = { cloud: "gemini", one: "vllm_adapter_c1", many: "vllm_adapter_c16" } as const;

function Tile({ big, what, how, id }: { big: string; what: string; how: string; id: string }) {
  return (
    <div data-testid={`tile-${id}`} className="flex flex-col gap-1.5 rounded-[10px] border border-line-soft bg-paper px-4 py-3.5">
      <span dir="ltr" className="self-start font-mono text-2xl font-semibold">{big}</span>
      <span className="text-sm leading-snug">{what}</span>
      <span className="text-xs text-muted">{how}</span>
    </div>
  );
}

export function ServingTiles({ serving, language }: { serving: Serving; language: Language }) {
  const { t } = useI18n();
  const arm = (id: string): Arm | undefined => serving.arms.find((a) => a.id === id);
  const cloud = arm(ARM.cloud);
  const one = arm(ARM.one);
  const many = arm(ARM.many);
  const tiles: { id: string; big: string; what: string; how: string }[] = [];

  if (one && many && one.throughput.calls_per_min > 0) {
    tiles.push({
      id: "throughput",
      big: `${(many.throughput.calls_per_min / one.throughput.calls_per_min).toFixed(1)}×`,
      what: t("results.tile.throughput", { high: many.throughput.concurrency, low: one.throughput.concurrency }),
      how: t("results.tile.throughputHow", { hardware: many.hardware, nHigh: many.throughput.calls, nLow: one.throughput.calls }),
    });
  }
  const manyCost = many?.[language].cost_per_1000_usd;
  const oneCost = one?.[language].cost_per_1000_usd;
  if (many && manyCost != null && oneCost != null) {
    tiles.push({
      id: "cost",
      big: `$${fixed(manyCost)}`,
      what: t("results.tile.cost", { high: many.throughput.concurrency, low: fixed(oneCost) }),
      how: t("results.tile.costHow", { spend: fixed(serving.actual_spend_usd, 0) }),
    });
  }
  const cell = cloud?.[language];
  if (cloud && cell && cell.latency_p50_s != null) {
    const excluded = cell.n - cell.latency_n;
    tiles.push({
      id: "latency",
      big: `${fixed(cell.latency_p50_s, 1)} s`,
      what: t("results.tile.latency", { p95: fixed(cell.latency_p95_s, 1) }),
      how:
        excluded > 0
          ? t("results.tile.latencyExcluded", { label: cloud.models.join(", "), n: cell.latency_n, excluded })
          : t("results.tile.latencyHow", { label: cloud.models.join(", "), n: cell.latency_n }),
    });
  }
  if (many) {
    const c = many[language];
    tiles.push({
      id: "failures",
      big: `${c.n - c.schema_valid} / ${c.n}`,
      what: t("results.tile.failures", { concurrency: many.throughput.concurrency }),
      how: many.label,
    });
  }

  return (
    <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      {tiles.map((tile) => (
        <Tile key={tile.id} {...tile} />
      ))}
    </div>
  );
}
