import type { Language } from "../api/client";

/**
 * Text direction of one transcript turn, from the API's per-turn language
 * (script-based: `TranscriptTurn.language`). Independent of the UI language: an
 * Arabic turn is RTL in the English UI, an English turn LTR in the Arabic UI.
 * `mixed` is `auto`, so the browser follows the turn's first strong character.
 */
export function turnDirection(language: Language): "rtl" | "ltr" | "auto" {
  if (language === "ar") return "rtl";
  if (language === "en") return "ltr";
  return "auto";
}

/**
 * Direction of free text with no language tag (rule text, conflict reasoning), by which script has
 * more letters. `dir="auto"` would follow the first strong character, so an English rule that
 * opens with a quoted Arabic word ("بكرا" sets the deadline…) would render right to left.
 */
export function textDirection(text: string): "rtl" | "ltr" {
  const arabic = text.match(/[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]/g)?.length ?? 0;
  const latin = text.match(/[A-Za-z]/g)?.length ?? 0;
  return arabic > latin ? "rtl" : "ltr";
}
