// Deterministic frame capture for the demo video (dev server must be running on :5173).
// Usage: node scripts/capture-frames.mjs <out-dir> <segments.json>
// Each segment: {replay, ghost?, t0, t1, fps, speed, camera, caption?, hudOnly?}
import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { chromium } from "@playwright/test";

const [outDir, segFile] = process.argv.slice(2);
const segments = JSON.parse(readFileSync(segFile, "utf8"));
mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({ args: ["--use-angle=metal", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });
let frame = 0;
async function caption(text) {
  await page.evaluate((text) => {
    let cap = document.getElementById("caption");
    if (!cap) {
      cap = document.createElement("div");
      cap.id = "caption";
      cap.style.cssText = "position:fixed;top:14px;left:50%;transform:translateX(-50%);padding:10px 18px;border-radius:10px;background:rgba(10,14,20,.8);color:#fff;font:600 20px system-ui;z-index:10;max-width:70vw;text-align:center";
      document.body.appendChild(cap);
    }
    cap.textContent = text ?? "";
    cap.style.display = text ? "block" : "none";
  }, text);
}

// Fleet view segment: {fleet, baseline, t0, t1, fps, speed, orbit: [heading0, heading1, pitch, range], caption}
async function captureFleet(seg) {
  await page.goto(`http://localhost:5173/?fleet=${encodeURIComponent(seg.fleet)}`);
  await page.waitForFunction(() => window.__fleet && window.__tileset, null, { timeout: 120000 });
  await caption(seg.caption);
  await page.evaluate((b) => {
    const c = window.__fleet.ctl.current;
    c.playing = false;
    c.baseline = b;
  }, seg.baseline);
  await page.getByRole("button", { name: seg.baseline === "B1" ? "B1 stock" : "B2 + contingency" }).click();
  const n = Math.round(((seg.t1 - seg.t0) / seg.speed) * seg.fps);
  const [h0, h1, pitch, range] = seg.orbit;
  for (let k = 0; k < n; k++) {
    const t = seg.t0 + (k * seg.speed) / seg.fps;
    const heading = h0 + ((h1 - h0) * k) / Math.max(n - 1, 1);
    await page.evaluate(
      async ({ t, heading, pitch, range }) => {
        const f = window.__fleet, v = window.__skyview, ts = window.__tileset;
        f.ctl.current.seek = t;
        f.scene.orbit(heading, pitch, range);
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        for (let i = 0; i < 40; i++) {
          v.scene.render();
          if (ts.tilesLoaded && i >= 2) break;
          await new Promise((r) => setTimeout(r, 50));
        }
        await new Promise((r) => setTimeout(r, 220)); // the panel's counts refresh at 5 Hz
      },
      { t, heading, pitch, range },
    );
    await page.screenshot({ path: join(outDir, `f${String(frame).padStart(5, "0")}.png`) });
    frame++;
    if (k % 50 === 0) console.log(`fleet ${seg.fleet} ${seg.baseline} t=${t.toFixed(1)} frame ${frame}`);
  }
}

for (const seg of segments) {
  if (seg.fleet) {
    await captureFleet(seg);
    continue;
  }
  const q = new URLSearchParams({ replay: seg.replay, ...(seg.ghost ? { ghost: seg.ghost } : {}) });
  await page.goto(`http://localhost:5173/?${q}`);
  await page.waitForFunction(() => window.__store?.getState().replay && window.__tileset, null, { timeout: 120000 });
  await page.evaluate(({ camera, caption }) => {
    const st = window.__store.getState();
    st.setPlaying(false);
    st.setCamera(camera);
    window.__controller.setReplays(st.replay, st.ghost);
    let cap = document.getElementById("caption");
    if (!cap) {
      cap = document.createElement("div");
      cap.id = "caption";
      cap.style.cssText = "position:fixed;top:14px;left:50%;transform:translateX(-50%);padding:10px 18px;border-radius:10px;background:rgba(10,14,20,.8);color:#fff;font:600 20px system-ui;z-index:10;max-width:70vw;text-align:center";
      document.body.appendChild(cap);
    }
    cap.textContent = caption ?? "";
    cap.style.display = caption ? "block" : "none";
  }, seg);
  const n = Math.round(((seg.t1 - seg.t0) / seg.speed) * seg.fps);
  for (let k = 0; k < n; k++) {
    const t = seg.t0 + (k * seg.speed) / seg.fps;
    const started = Date.now();
    await page.evaluate(async (t) => {
      const st = window.__store.getState();
      st.setTime(t);
      const v = window.__skyview, c = window.__controller, ts = window.__tileset;
      const s = window.__store.getState();
      for (let i = 0; i < 40; i++) {
        c.update(t, s.layers, s.camera);
        v.scene.render();
        if (ts.tilesLoaded && i >= 2) break;
        await new Promise((r) => setTimeout(r, 50));
      }
    }, t);
    await page.screenshot({ path: join(outDir, `f${String(frame).padStart(5, "0")}.png`) });
    frame++;
    if (k % 50 === 0) console.log(`${seg.replay} t=${t.toFixed(1)} frame ${frame} (${Date.now() - started} ms)`);
  }
}
await browser.close();
console.log(`${frame} frames`);
