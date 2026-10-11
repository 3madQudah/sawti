import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { dictionaries, format, type StringKey, type UiLanguage } from "./strings";

const STORAGE_KEY = "sawti.ui-language";

interface I18n {
  lang: UiLanguage;
  dir: "ltr" | "rtl";
  setLang: (lang: UiLanguage) => void;
  t: (key: StringKey, vars?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18n | null>(null);

function initialLanguage(): UiLanguage {
  try {
    return localStorage.getItem(STORAGE_KEY) === "ar" ? "ar" : "en";
  } catch {
    return "en";
  }
}

/**
 * UI language: English by default, Arabic as a toggle that switches the whole
 * page to RTL (`<html dir lang>`). Transcript turns set their own direction
 * independently (see `lib/direction.ts`).
 */
export function I18nProvider({ children, initial }: { children: ReactNode; initial?: UiLanguage }) {
  const [lang, setLangState] = useState<UiLanguage>(initial ?? initialLanguage);
  const dir = lang === "ar" ? "rtl" : "ltr";

  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = dir;
  }, [lang, dir]);

  const setLang = useCallback((next: UiLanguage) => {
    setLangState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Storage unavailable (private window): the choice lasts for this page only.
    }
  }, []);

  const value = useMemo<I18n>(
    () => ({ lang, dir, setLang, t: (key, vars) => format(dictionaries[lang][key], vars) }),
    [lang, dir, setLang],
  );
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18n {
  const context = useContext(I18nContext);
  if (context === null) throw new Error("useI18n outside I18nProvider");
  return context;
}
