// A fly-by video of a world in film mode (docs/07 §15): renders each camera of a path from
// tools/flyby_path.py, with the clock (turbines, water) stepped frame by
// frame, then encodes them with ffmpeg. The dev server must be on :5173 and a live server
// serving the world (default ws://localhost:8765).
// Usage: node scripts/flyby.mjs <frames.json> <out.mp4> <ffmpeg> [live ws url] [width] [height]
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { chromium } from "@playwright/test";

const [frameFile, out, ffmpeg, live = "ws://localhost:8765", w = "1920", h = "1080"] = process.argv.slice(2);
const frames = JSON.parse(readFileSync(frameFile, "utf8"));
const dir = join(dirname(out), "flyby-frames");
rmSync(dir, { recursive: true, force: true });
mkdirSync(dir, { recursive: true });
const browser = await chromium.launch({ args: ["--use-angle=metal", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: Number(w), height: Number(h) }, deviceScaleFactor: 1 });
await page.goto(`http://localhost:5173/?live=${live}&film=1`);
await page.waitForFunction(() => window.__view?.world?.trees?.length > 0, null, { timeout: 120000 });
await page.waitForTimeout(3000); // textures and the first shadows
const fps = frames.length > 1 ? 1 / (frames[1].t - frames[0].t) : 30;
for (let k = 0; k < frames.length; k++) {
  const f = frames[k];
  await page.evaluate(
    ([t, eye, at]) =>
      new Promise((done) => {
        window.__view.setClock(t);
        window.__view.setShot(eye, at);
        requestAnimationFrame(() => requestAnimationFrame(() => done(null)));
      }),
    [f.t, f.eye, f.at],
  );
  await page.screenshot({ path: join(dir, `${String(k).padStart(5, "0")}.jpg`), type: "jpeg", quality: 92 });
  if (k % 150 === 0) console.log(`frame ${k}/${frames.length}`);
}
await browser.close();
execFileSync(ffmpeg, ["-y", "-loglevel", "error", "-framerate", String(Math.round(fps)), "-i", join(dir, "%05d.jpg"),
  "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]); // prettier-ignore
console.log(out);
