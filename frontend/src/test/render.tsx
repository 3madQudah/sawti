import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { vi } from "vitest";

import { App } from "../App";
import { I18nProvider } from "../i18n/I18nProvider";
import type { UiLanguage } from "../i18n/strings";
import { MockApi } from "./mockApi";

/** Render the whole app at `path` against a fresh mock API. */
export function renderApp(path: string, { lang = "en" as UiLanguage, api = new MockApi() } = {}) {
  vi.stubGlobal("fetch", api.fetch);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const ui: ReactElement = (
    <QueryClientProvider client={client}>
      <I18nProvider initial={lang}>
        <MemoryRouter initialEntries={[path]}>
          <Routes>
            <Route path="*" element={<App />} />
          </Routes>
        </MemoryRouter>
      </I18nProvider>
    </QueryClientProvider>
  );
  return { api, ...render(ui) };
}
