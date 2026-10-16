import { defineConfig } from "@playwright/test";

// The demo recording. Not part of `npm run e2e`: its own config, its own folder.
export default defineConfig({
  testDir: ".",
  testMatch: /demo\.spec\.ts/,
  workers: 1,
  reporter: [["line"]],
  outputDir: "out/test-results",
  use: {
    baseURL: "http://localhost:5175",
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
    launchOptions: process.env.DEMO_CHROMIUM ? { executablePath: process.env.DEMO_CHROMIUM } : {},
  },
  webServer: {
    command: "npx vite --port 5175 --strictPort",
    cwd: "..",
    url: "http://localhost:5175",
    reuseExistingServer: false,
    env: { VITE_SAWTI_API_KEY: "demo-key", VITE_SAWTI_REVIEWER_ID: "qa-emad" },
  },
});
