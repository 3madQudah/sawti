import { useRef, useState, type DragEvent } from "react";
import { Link } from "react-router-dom";

import { ApiError, LANGUAGES, type Language } from "../../api/client";
import { RUNNING, useCall, useUploadAudio } from "../../api/hooks";
import { LangChip } from "../../components/LangChip";
import { useI18n } from "../../i18n/I18nProvider";
import { fixed, shortId } from "../../lib/format";
import { Pipeline } from "./Pipeline";
import { loadRecent, rememberUpload, type RecentUpload } from "./recent";

/** The API's limits (`POST /calls/audio`); checked here first only to fail fast. The server is the authority. */
const ACCEPTED = [".wav", ".mp3", ".m4a"];
const MAX_MB = 50;
/** Code-switched first: it is the case auto-detection gets wrong. */
const LANGUAGE_ORDER: readonly Language[] = ["mixed", ...LANGUAGES.filter((l) => l !== "mixed")];

type FileProblem = "type" | "size" | "empty";

function check(file: File): FileProblem | null {
  const suffix = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!ACCEPTED.includes(suffix)) return "type";
  if (file.size === 0) return "empty";
  if (file.size > MAX_MB * 1024 * 1024) return "size";
  return null;
}

export function NewCallPage() {
  const { t } = useI18n();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [problem, setProblem] = useState<FileProblem | null>(null);
  const [language, setLanguage] = useState<Language | null>(null);
  const [agent, setAgent] = useState("");
  const [callId, setCallId] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentUpload[]>(loadRecent);
  const upload = useUploadAudio();
  const current = useCall(callId ?? "", { pollWhileRunning: true });
  const running = callId !== null && (current.data === undefined || RUNNING.includes(current.data.status));

  const pick = (picked: File | undefined) => {
    if (!picked) return;
    setFile(picked);
    setProblem(check(picked));
    upload.reset();
  };
  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    pick(event.dataTransfer.files[0]);
  };
  const start = () => {
    if (!file || !language || problem) return;
    upload.mutate(
      { file, language, agentExternalId: agent },
      {
        onSuccess: ({ call_id }) => {
          setCallId(call_id);
          setRecent(rememberUpload({ callId: call_id, language, at: new Date().toISOString() }));
        },
      },
    );
  };
  const again = () => {
    setFile(null);
    setProblem(null);
    setCallId(null);
    upload.reset();
    if (input.current) input.current.value = "";
  };

  const serverError = upload.error instanceof ApiError ? upload.error : null;
  const label = upload.isPending ? t("newCall.uploading") : running ? t("newCall.processing") : t("newCall.start");

  return (
    <main className="mx-auto flex w-full max-w-[1280px] flex-col gap-6 p-8">
      <div className="flex max-w-[760px] flex-col gap-1.5">
        <h1 className="m-0 text-[28px] font-semibold">{t("newCall.title")}</h1>
        <p className="m-0 leading-relaxed text-muted">{t("newCall.subtitle")}</p>
      </div>

      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-5">
        <section className="flex flex-col gap-5 rounded-[10px] border border-line bg-white p-6">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            className="flex flex-col items-center gap-2.5 rounded-[10px] border-2 border-dashed border-field bg-paper px-6 py-8 text-center"
          >
            <span className="font-semibold">{t("newCall.drop")}</span>
            <span dir="auto" className="text-[13px] text-muted">{t("newCall.dropHint", { mb: MAX_MB })}</span>
            <button type="button" onClick={() => input.current?.click()} className="min-h-10 rounded-lg border border-field bg-white px-4 font-medium">
              {t("newCall.browse")}
            </button>
            <input
              ref={input}
              type="file"
              accept={ACCEPTED.join(",")}
              aria-label={t("newCall.fileInput")}
              className="sr-only"
              onChange={(e) => pick(e.target.files?.[0])}
            />
          </div>

          {file && (
            <div className="flex items-center justify-between gap-3 rounded-lg border border-line px-4 py-3">
              <span className="flex min-w-0 flex-col">
                <span dir="ltr" data-testid="file-name" className="truncate text-start font-mono text-sm">{file.name}</span>
                <span className="text-xs text-muted">{t("newCall.size", { size: fixed(file.size / (1024 * 1024), 1) })}</span>
              </span>
              <button type="button" onClick={again} disabled={running || upload.isPending} className="min-h-9 rounded-lg border border-field px-3 text-sm">
                {t("newCall.remove")}
              </button>
            </div>
          )}
          {problem && (
            <p role="alert" className="-mt-2 m-0 text-sm text-danger">
              {problem === "type" ? t("newCall.error.type") : problem === "empty" ? t("newCall.error.empty") : t("newCall.error.size", { mb: MAX_MB })}
            </p>
          )}

          <fieldset className="m-0 flex flex-col gap-2 border-0 p-0">
            <legend className="mb-2 font-medium">{t("newCall.language")}</legend>
            <div className="flex flex-wrap gap-2">
              {LANGUAGE_ORDER.map((l) => (
                <label key={l} className={`flex min-h-10 cursor-pointer items-center gap-2 rounded-lg border px-3.5 ${language === l ? "border-ink" : "border-field"}`}>
                  <input type="radio" name="language" value={l} checked={language === l} onChange={() => setLanguage(l)} />
                  <span>{t(`lang.${l}`)}</span>
                  <LangChip language={l} />
                </label>
              ))}
            </div>
            <span className="text-[13px] text-muted">{t("newCall.languageHint")}</span>
          </fieldset>

          <label className="flex flex-col gap-1.5">
            <span className="font-medium">{t("newCall.agent")}</span>
            <input
              dir="ltr"
              value={agent}
              maxLength={255}
              onChange={(e) => setAgent(e.target.value)}
              className="min-h-10 rounded-lg border border-field px-3 font-mono text-sm"
            />
            <span className="text-[13px] text-muted">{t("newCall.agentHint")}</span>
          </label>

          {upload.error && (
            <p role="alert" className="m-0 text-sm text-danger">
              {serverError
                ? t("newCall.error.server", { status: serverError.status, message: serverError.message })
                : t("newCall.error.network", { message: upload.error.message })}
            </p>
          )}
          {callId && !running ? (
            <button type="button" onClick={again} className="min-h-12 rounded-[10px] bg-ink text-[15px] font-semibold text-white">
              {t("newCall.again")}
            </button>
          ) : (
            <button
              type="button"
              onClick={start}
              disabled={!file || !language || problem !== null || upload.isPending || running}
              className="min-h-12 rounded-[10px] bg-ink text-[15px] font-semibold text-white disabled:cursor-not-allowed disabled:bg-muted"
            >
              {label}
            </button>
          )}
          <p className="m-0 text-[13px] leading-relaxed text-muted">{t("newCall.privacy")}</p>
        </section>

        <div className="flex flex-col gap-5">
          <section aria-labelledby="pipeline-title" className="flex flex-col gap-3 rounded-[10px] border border-line bg-white p-6">
            <h2 id="pipeline-title" className="m-0 text-lg font-semibold">{t("newCall.pipeline")}</h2>
            {callId ? <Pipeline callId={callId} /> : <p className="m-0 text-muted">{t("newCall.pipelineIdle")}</p>}
          </section>

          <section aria-labelledby="recent-title" className="flex flex-col gap-2 rounded-[10px] border border-line bg-white p-6">
            <div className="flex items-baseline justify-between">
              <h2 id="recent-title" className="m-0 text-lg font-semibold">{t("newCall.recent")}</h2>
              <span className="text-xs text-muted">{t("newCall.recentNote")}</span>
            </div>
            {recent.length === 0 ? (
              <p className="m-0 text-sm text-muted">{t("newCall.recentEmpty")}</p>
            ) : (
              <ul className="m-0 list-none p-0">
                {recent.map((r) => (
                  <RecentRow key={r.callId} upload={r} />
                ))}
              </ul>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}

const STATUS_STYLE: Record<string, string> = {
  awaiting_review: "bg-amber-tint text-amber-text",
  completed: "bg-teal-tint text-teal-dark",
  reviewed: "bg-teal-tint text-teal-dark",
  failed: "bg-danger/10 text-danger",
};

function RecentRow({ upload }: { upload: RecentUpload }) {
  const { t } = useI18n();
  const call = useCall(upload.callId, { pollWhileRunning: true });
  const status = call.data?.status;
  return (
    <li className="flex items-center justify-between gap-3 border-b border-line-soft py-2.5 last:border-b-0">
      <span className="flex items-center gap-2.5">
        <Link dir="ltr" to={`/calls/${upload.callId}`} className="font-mono text-sm" title={upload.callId}>
          {shortId(upload.callId)}
        </Link>
        <LangChip language={upload.language} />
        <span dir="ltr" className="font-mono text-xs text-muted">{new Date(upload.at).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" })}</span>
      </span>
      {status && (
        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${STATUS_STYLE[status] ?? "bg-rubric text-subtle"}`}>
          {t(`status.${status}`)}
        </span>
      )}
    </li>
  );
}
