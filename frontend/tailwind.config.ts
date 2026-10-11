import type { Config } from "tailwindcss";

// Colours and type from the approved mockups (docs/design/*.dc.html).
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#15171C",
        paper: "#F5F6F4",
        line: "#E2E5E1",
        "line-soft": "#EEF0ED",
        muted: "#5B616B",
        subtle: "#3A3F47",
        field: "#C9CEC8",
        teal: { DEFAULT: "#0E6B66", dark: "#094A47", soft: "#CDE7E4", tint: "#E6F2F0", mint: "#9FD3CF" },
        amber: { DEFAULT: "#B45309", text: "#7A3A04", tint: "#FBF4E8", edge: "#EBD3AE", chip: "#FCEFD9" },
        danger: { DEFAULT: "#8A2A1C", quote: "#8A3B12" },
        nav: { bg: "#2A2E36", text: "#B9BEC6" },
        mark: { selected: "#FFD98A", other: "#DCEDEA" },
        rubric: "#ECEEEB",
        customer: "#F8F9F7",
      },
      fontFamily: {
        sans: ["'IBM Plex Sans'", "'IBM Plex Sans Arabic'", "system-ui", "sans-serif"],
        arabic: ["'IBM Plex Sans Arabic'", "'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
      minWidth: { app: "1024px" },
    },
  },
  plugins: [],
} satisfies Config;
