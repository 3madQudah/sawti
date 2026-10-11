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
