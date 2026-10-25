// The explorer's build. `npm run dev` proxies the API to `hcsim app serve` on its default port.
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  test: { environment: "jsdom" },
});
