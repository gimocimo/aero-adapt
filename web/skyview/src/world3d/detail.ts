// Ground detail (docs/07 §15): tileable grey textures that give each kind of land its grain
// (grass, crops in rows, ploughed soil, tarmac, a wood's floor), made here from noise so that
// nothing need be downloaded, and a water normal map. The land's colour comes from the
// painted cover; these only modulate it, up close.
import * as THREE from "three";

const N = 512;

/** Seeded, tileable value noise (period ``p`` cells across the texture) as a function of
 * pixel (x, y). */
function valueNoise(p: number, seed: number): (x: number, y: number) => number {
  const grid = new Float32Array(p * p);
  let s = seed * 9301 + 49297;
  for (let i = 0; i < grid.length; i++) {
    s = (s * 9301 + 49297) % 233280;
    grid[i] = s / 233280;
  }
  const at = (i: number, j: number) => grid[((i % p) + p) % p * p + (((j % p) + p) % p)]!;
  return (x: number, y: number) => {
    const fx = (x / N) * p;
    const fy = (y / N) * p;
    const i = Math.floor(fx);
    const j = Math.floor(fy);
    const u = fx - i;
    const v = fy - j;
    const su = u * u * (3 - 2 * u);
    const sv = v * v * (3 - 2 * v);
    const a = at(i, j) * (1 - su) + at(i + 1, j) * su;
    const b = at(i, j + 1) * (1 - su) + at(i + 1, j + 1) * su;
    return a * (1 - sv) + b * sv;
  };
}

function fbm(octaves: [number, number][], seed: number): (x: number, y: number) => number {
  const ns = octaves.map(([p], k) => valueNoise(p, seed + k * 17));
  const total = octaves.reduce((t, [, a]) => t + a, 0);
  return (x, y) => ns.reduce((acc, n, k) => acc + octaves[k]![1] * n(x, y), 0) / total;
}

function texture(fn: (x: number, y: number) => number): THREE.DataTexture {
  const data = new Uint8Array(N * N * 4);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const v = Math.max(0, Math.min(255, Math.round(255 * fn(x, y))));
      const k = 4 * (y * N + x);
      data[k] = data[k + 1] = data[k + 2] = v;
      data[k + 3] = 255;
    }
  }
  const t = new THREE.DataTexture(data, N, N, THREE.RGBAFormat);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}

export interface Detail {
  grass: THREE.DataTexture;
  crop: THREE.DataTexture;
  soil: THREE.DataTexture;
  paved: THREE.DataTexture;
  forest: THREE.DataTexture;
  waterNormal: THREE.DataTexture;
}

let cached: Detail | null = null;

/** The detail textures (made once per page). */
export function detailTextures(): Detail {
  if (cached) return cached;
  const fine = fbm([[96, 1], [192, 0.7], [64, 0.5], [16, 0.35]], 1);
  const tufts = fbm([[32, 1], [8, 0.6]], 7);
  const grass = texture((x, y) => 0.5 + 0.55 * (fine(x, y) - 0.5) + 0.35 * (tufts(x, y) - 0.5));
  const rowsNoise = fbm([[128, 1], [32, 0.5]], 3);
  const crop = texture((x, y) => {
    const row = 0.5 + 0.5 * Math.sin((y / N) * 2 * Math.PI * 24); // rows across the tile
    return 0.5 + 0.28 * (row - 0.5) + 0.45 * (rowsNoise(x, y) - 0.5);
  });
  const clods = fbm([[64, 1], [128, 0.8], [256, 0.5], [16, 0.5]], 11);
  const furrow = fbm([[24, 1]], 5);
  const soil = texture((x, y) => {
    const f = 0.5 + 0.5 * Math.sin((y / N) * 2 * Math.PI * 12 + 2.5 * furrow(x, y));
    return 0.5 + 0.7 * (clods(x, y) - 0.5) + 0.25 * (f - 0.5);
  });
  const speck = fbm([[256, 1], [128, 0.6]], 13);
  const stain = fbm([[8, 1], [4, 0.7]], 19);
  const paved = texture((x, y) => 0.5 + 0.35 * (speck(x, y) - 0.5) + 0.3 * (stain(x, y) - 0.5));
  const litter = fbm([[48, 1], [96, 0.7], [12, 0.6]], 23);
  const forest = texture((x, y) => 0.42 + 0.8 * (litter(x, y) - 0.5));
  // Water: a normal map from a gently rippled height.
  const h = fbm([[16, 1], [32, 0.6], [64, 0.35]], 29);
  const data = new Uint8Array(N * N * 4);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const dx = h(x + 1, y) - h(x - 1, y);
      const dy = h(x, y + 1) - h(x, y - 1);
      const n = new THREE.Vector3(-dx * 18, -dy * 18, 1).normalize();
      const k = 4 * (y * N + x);
      data[k] = Math.round((n.x * 0.5 + 0.5) * 255);
      data[k + 1] = Math.round((n.y * 0.5 + 0.5) * 255);
      data[k + 2] = Math.round((n.z * 0.5 + 0.5) * 255);
      data[k + 3] = 255;
    }
  }
  const waterNormal = new THREE.DataTexture(data, N, N, THREE.RGBAFormat);
  waterNormal.wrapS = waterNormal.wrapT = THREE.RepeatWrapping;
  waterNormal.minFilter = THREE.LinearMipmapLinearFilter;
  waterNormal.generateMipmaps = true;
  waterNormal.needsUpdate = true;
  cached = { grass, crop, soil, paved, forest, waterNormal };
  return cached;
}
