import type { Language } from "../api/client";

const STYLES: Record<Language, string> = {
  ar: "bg-ink text-white",
  mixed: "bg-teal-soft text-teal-dark",
  en: "bg-white text-ink ring-1 ring-inset ring-[#9AA0A8]",
};

/** The `ar` / `en` / `mixed` code chip. A code, not a word: it reads the same in both UI languages. */
export function LangChip({ language }: { language: Language }) {
  return (
    <span
      dir="ltr"
      className={`inline-block rounded-full px-[9px] py-[3px] font-mono text-xs font-semibold ${STYLES[language]}`}
    >
      {language}
    </span>
  );
}
