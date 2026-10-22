// Stills of a world in the cinematic look (docs/07 §15). The dev server must be running on
// :5173 and a live server on :8765 (the world comes from it).
// Usage: node scripts/stills.mjs <out-dir> <shots.json>
// Each shot: {name, eye: [n, e, h], at: [n, e, h]} (metres; heights above the datum).
import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "@playwright/test";

const [outDir, shotFile] = process.argv.slice(2);
const shots = JSON.parse(readFileSync(shotFile, "utf8"));
mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ args: ["--use-angle=metal", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto("http://localhost:5173/?live=ws://localhost:8765&film=1");
await page.waitForFunction(() => window.__view?.world?.trees?.length > 0, null, { timeout: 120000 });
for (const s of shots) {
  await page.evaluate(([eye, at]) => window.__view.setShot(eye, at), [s.eye, s.at]);
  await page.waitForTimeout(1500); // the near trees and shadows settle
  await page.screenshot({ path: join(outDir, `${s.name}.jpg`), type: "jpeg", quality: 88 });
  console.log(s.name);
}
await browser.close();
