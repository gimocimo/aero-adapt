import { defineConfig } from "@playwright/test";

// Smoke test (spec §11): tiles disabled, so it runs in CI without a Cesium ion token or a GPU.
// The live test (M2 D10) also needs the Python live server (uv, `live` extra).
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  use: {
    baseURL: "http://localhost:5174",
    launchOptions: { args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] },
  },
  webServer: [
    {
      command: "npm run dev -- --port 5174 --strictPort",
      url: "http://localhost:5174",
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
    {
      // Listens once the closed loop is compiled.
      command: "uv run aeroadapt serve --port 8799",
      cwd: "../..",
      port: 8799,
      reuseExistingServer: !process.env.CI,
      timeout: 300_000,
    },
  ],
});
