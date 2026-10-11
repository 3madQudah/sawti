import type { Language } from "../../api/client";

/** Calls uploaded from this browser. A per-viewer convenience: the API has no "list calls" endpoint. */
export interface RecentUpload {
  callId: string;
  language: Language;
  at: string;
}

const KEY = "sawti.recent-uploads";
const LIMIT = 5;

export function loadRecent(): RecentUpload[] {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as RecentUpload[]) : [];
  } catch {
    return [];
  }
}

export function rememberUpload(upload: RecentUpload): RecentUpload[] {
  const next = [upload, ...loadRecent().filter((u) => u.callId !== upload.callId)].slice(0, LIMIT);
  try {
    sessionStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // Storage blocked: the list just doesn't survive a reload.
  }
  return next;
}
