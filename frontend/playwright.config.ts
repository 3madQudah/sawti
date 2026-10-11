import { defineConfig, devices } from "@playwright/test";

// e2e against a mocked API: every `/api/**` request is answered in the
// browser by `e2e/mockApi.ts` from typed fixtures. No backend, no ASR, no LLM.
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5174",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: {
    command: "npx vite --port 5174 --strictPort",
    url: "http://localhost:5174",
    reuseExistingServer: false,
    env: { VITE_SAWTI_API_KEY: "e2e-key", VITE_SAWTI_REVIEWER_ID: "qa-emad" },
  },
});
