import { createReadStream, cpSync, existsSync, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";
import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

// CesiumJS needs its static Workers/Assets/Widgets/ThirdParty folders served at runtime.
// This small plugin serves them in dev and copies them at build time. It replaces
// vite-plugin-static-copy, which pulled in a vulnerable transitive dependency.
const CESIUM_BUILD = resolve(import.meta.dirname, "node_modules/cesium/Build/Cesium");
const CESIUM_BASE = "cesiumStatic";
const CESIUM_DIRS = ["Workers", "ThirdParty", "Assets", "Widgets"];
const MIME: Record<string, string> = {
  ".js": "text/javascript",
  ".json": "application/json",
  ".css": "text/css",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".xml": "application/xml",
  ".wasm": "application/wasm",
  ".glb": "model/gltf-binary",
  ".ktx2": "image/ktx2",
};

function cesiumStatic(): Plugin {
  return {
    name: "aeroadapt-cesium-static",
    configureServer(server) {
      server.middlewares.use(`/${CESIUM_BASE}`, (req, res, next) => {
        const rel = normalize(decodeURIComponent((req.url ?? "/").split("?")[0]));
        const file = join(CESIUM_BUILD, rel);
        if (!file.startsWith(CESIUM_BUILD) || !existsSync(file) || !statSync(file).isFile()) {
          next();
          return;
        }
        res.setHeader("Content-Type", MIME[extname(file)] ?? "application/octet-stream");
        createReadStream(file).pipe(res);
      });
    },
    writeBundle(options) {
      const outDir = options.dir ?? resolve(import.meta.dirname, "dist");
      for (const dir of CESIUM_DIRS) {
        cpSync(join(CESIUM_BUILD, dir), join(outDir, CESIUM_BASE, dir), { recursive: true });
      }
    },
  };
}

export default defineConfig({
  define: { CESIUM_BASE_URL: JSON.stringify(`/${CESIUM_BASE}`) },
  plugins: [react(), cesiumStatic()],
  // Pre-bundle the live client's codec: discovered at runtime instead, it makes a cold dev
  // server reload the page once (the live connection then drops with 1001).
  optimizeDeps: { include: ["@msgpack/msgpack"] },
  // CesiumJS alone is ~4 MB minified (~1.2 MB gzip); code-splitting it buys little here.
  build: { chunkSizeWarningLimit: 5000 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
