import { NavLink } from "react-router-dom";

import { reviewerId } from "../api/client";
import { useI18n } from "../i18n/I18nProvider";
import type { StringKey } from "../i18n/strings";

const NAV: readonly { to: string; label: StringKey }[] = [
  { to: "/", label: "nav.queue" },
  { to: "/new", label: "nav.newCall" },
  { to: "/memory", label: "nav.memory" },
  { to: "/results", label: "nav.results" },
];

export function Header() {
  const { t, lang, setLang } = useI18n();
  const toggle = (on: boolean) =>
    `min-h-8 min-w-11 rounded-md px-2.5 text-[13px] font-semibold ${on ? "bg-paper text-ink" : "bg-transparent text-nav-text"}`;
  return (
    <header className="flex min-h-[60px] flex-wrap items-center justify-between gap-4 bg-ink px-8 text-paper">
      <div className="flex flex-wrap items-center gap-7">
        <div className="flex items-baseline gap-2.5">
          <span className="font-sans text-xl font-semibold tracking-tight">{t("app.name")}</span>
          <span className="font-arabic text-lg text-teal-mint">صوتي</span>
        </div>
        <nav className="flex flex-wrap gap-1">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `rounded-md px-3.5 py-2.5 no-underline ${isActive ? "bg-nav-bg font-medium text-white hover:text-white" : "text-nav-text hover:text-white"}`
              }
            >
              {t(item.label)}
            </NavLink>
          ))}
        </nav>
      </div>
      <div className="flex items-center gap-3 text-nav-text">
        <div role="group" aria-label={t("header.uiLanguage")} className="flex gap-0.5 rounded-lg bg-nav-bg p-[3px]">
          <button type="button" aria-pressed={lang === "en"} className={toggle(lang === "en")} onClick={() => setLang("en")}>
            EN
          </button>
          <button type="button" aria-pressed={lang === "ar"} className={`${toggle(lang === "ar")} font-arabic`} onClick={() => setLang("ar")}>
            عربي
          </button>
        </div>
        <span>{t("header.reviewer")}</span>
        <span dir="ltr" className="rounded-md bg-nav-bg px-2.5 py-1.5 font-mono text-white">
          {reviewerId}
        </span>
      </div>
    </header>
  );
}
