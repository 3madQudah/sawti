import { useI18n } from "../i18n/I18nProvider";

export function Loading() {
  const { t } = useI18n();
  return <p className="text-muted" role="status">{t("common.loading")}</p>;
}

export function LoadError({ error, retry }: { error: unknown; retry: () => void }) {
  const { t } = useI18n();
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className="flex items-center gap-3 rounded-[10px] border border-danger/30 bg-white px-5 py-4">
      <span>{t("common.error", { message })}</span>
      <button type="button" onClick={retry} className="min-h-10 rounded-lg border border-field px-3.5 font-medium">
        {t("common.retry")}
      </button>
    </div>
  );
}
