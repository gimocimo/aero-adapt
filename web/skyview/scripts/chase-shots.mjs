// Chase-view stills of a live flight in film mode, at chosen simulation times (docs/07 §15).
// Usage: node scripts/chase-shots.mjs <out-dir> <live ws url> <t1,t2,...> [speed]
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "@playwright/test";

const [outDir, live, times, speed = "4"] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ args: ["--use-angle=metal", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto(`http://localhost:5173/?live=${live}&film=1`);
await page.waitForFunction(() => window.__view?.world?.trees?.length > 0 && window.__store?.getState().live?.connected, null, { timeout: 120000 });
await page.evaluate((x) => window.__store.getState().liveSend({ cmd: "set_speed", x }), Number(speed));
for (const t of times.split(",").map(Number)) {
  // The store's ``time`` is where playback is (``live.timeS`` is only the last status).
  await page.waitForFunction((t) => window.__store.getState().time >= t, t, { timeout: 600000, polling: 200 });
  await page.screenshot({ path: join(outDir, `chase-${String(Math.round(t)).padStart(4, "0")}.jpg`), type: "jpeg", quality: 88 });
  console.log(t, await page.evaluate(() => window.__view.chaseScale));
}
await browser.close();
