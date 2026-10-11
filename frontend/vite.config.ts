import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The dashboard calls `/api/...`; the dev server forwards it to the local API
// (docker compose `api`, port 8000), so the browser never needs CORS and
// nothing leaves localhost. See docs/DEMO.md.
const apiTarget = process.env.SAWTI_API_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": { target: apiTarget, rewrite: (path) => path.replace(/^\/api/, "") } },
  },
  preview: {
    port: 4173,
    strictPort: true,
    proxy: { "/api": { target: apiTarget, rewrite: (path) => path.replace(/^\/api/, "") } },
  },
});
