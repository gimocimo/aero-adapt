// Materials for the cinematic look (docs/07 §15): the ground with its grain by land cover,
// building facades (windows by floor; glass curtain walls on towers) and roof tiles, made in
// the shader from each surface's own coordinates, and water that reflects the sky.
import * as THREE from "three";
import { detailTextures } from "./detail";
import { CLASS_STEP } from "./paint";

/** The ground: the painted cover, with photo-like grain by class up close and a broad
 * mottling far off (so a field is never one flat colour). */
export function groundMaterial(map: THREE.Texture, classes: THREE.Texture): THREE.MeshLambertMaterial {
  const d = detailTextures();
  // Matte (Lambert): a physically based ground would catch the sky's sheen at the grazing
  // angles a chase camera sees it from, and turn every field grey.
  const mat = new THREE.MeshLambertMaterial({ map });
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, {
      uClass: { value: classes },
      uGrass: { value: d.grass },
      uCrop: { value: d.crop },
      uSoil: { value: d.soil },
      uPaved: { value: d.paved },
      uForest: { value: d.forest },
      uMottle: { value: d.mottle },
    });
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vWorld;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;");
    sh.fragmentShader = sh.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vWorld;
uniform sampler2D uClass, uGrass, uCrop, uSoil, uPaved, uForest, uMottle;`,
      )
      .replace(
        "#include <map_fragment>",
        `#include <map_fragment>
{
  float cls = floor(texture2D(uClass, vMapUv).r * 255.0 / ${CLASS_STEP.toFixed(1)} + 0.5);
  vec2 w = vec2(vWorld.x, -vWorld.z);
  float dist = length(vWorld - cameraPosition);
  float near = 1.0 - smoothstep(180.0, 1600.0, dist);
  vec3 g = vec3(0.5);
  if (cls < 0.5) g = 0.6 * texture2D(uGrass, w / 5.0).rgb + 0.4 * texture2D(uGrass, w / 23.0).rgb;
  else if (cls < 1.5) g = 0.6 * texture2D(uCrop, w / 6.0).rgb + 0.4 * texture2D(uMottle, w / 29.0).rgb;
  else if (cls < 2.5) g = 0.65 * texture2D(uSoil, w / 20.0).rgb + 0.35 * texture2D(uSoil, w / 7.0).rgb;
  else if (cls < 3.5) g = 0.7 * texture2D(uPaved, w / 30.0).rgb + 0.3 * texture2D(uPaved, w / 11.0).rgb;
  else if (cls < 4.5) g = 0.6 * texture2D(uForest, w / 4.0).rgb + 0.4 * texture2D(uForest, w / 19.0).rgb;
  float strength = cls > 4.5 ? 0.0 : (cls > 2.5 && cls < 3.5 ? 1.0 : 1.6);
  diffuseColor.rgb *= 1.0 + near * strength * (g - 0.5);
  // Patches a few tens of metres across (thinner and lusher grass, wetter soil), out to a few
  // kilometres, and broad mottling everywhere: no field is one flat colour.
  float mid = 0.6 * texture2D(uMottle, w / 70.0).r + 0.4 * texture2D(uMottle, w / 150.0).g;
  float midNear = 1.0 - smoothstep(400.0, 3500.0, dist);
  diffuseColor.rgb *= 1.0 + (cls < 2.5 ? 1.3 : 0.5) * midNear * (mid - 0.5);
  float broad = texture2D(uMottle, w / 260.0).r * 0.6 + texture2D(uMottle, w / 900.0).g * 0.4;
  diffuseColor.rgb *= 1.0 + 0.6 * (broad - 0.5);
  // A map's colours are brighter than land's: take a fifth of the saturation out.
  diffuseColor.rgb = mix(diffuseColor.rgb, vec3(dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114))), 0.22);
}`,
      );
  };
  return mat;
}

/** Facade and roof kinds, per vertex (``aFacade.z``). */
export const FACADE = { none: 0, homes: 1, offices: 2, glass: 3, industry: 4, tiles: 5, flat: 6, stone: 7, civic: 8 } as const;

/** Buildings: vertex colours (walls and roofs) with windows, glass and tiles drawn from each
 * surface's coordinates (``aFacade``: metres along, metres up, kind). */
export function buildingMaterial(): THREE.MeshStandardMaterial {
  const d = detailTextures();
  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.82,
    metalness: 0,
    flatShading: true,
    side: THREE.DoubleSide,
  });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uNoise = { value: d.grain };
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nattribute vec3 aFacade;\nvarying vec3 vFacade;\nvarying vec3 vSpot;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvFacade = aFacade;\nvSpot = (modelMatrix * vec4(transformed, 1.0)).xyz;");
    sh.fragmentShader = sh.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vFacade;
varying vec3 vSpot;
uniform sampler2D uNoise;
float vFacadeGlass = 0.0;
float hash2(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
// A soft-edged box pattern (antialiased with screen-space derivatives).
float boxes(vec2 p, vec2 period, vec2 size) {
  vec2 q = abs(fract(p / period) - 0.5) * period;
  vec2 fw = fwidth(p) + 1e-4;
  vec2 m = 1.0 - smoothstep(size * 0.5 - fw, size * 0.5 + fw, q);
  float fade = 1.0 - smoothstep(0.35, 0.8, max(fw.x / period.x, fw.y / period.y));
  return m.x * m.y * fade;
}`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
{
  float kind = floor(vFacade.z + 0.5);
  vec2 uv = vFacade.xy;
  float grain = texture2D(uNoise, uv / 6.0).r;
  // Each building its own scatter of panes (a seed from where it stands), and its own shade:
  // a little lighter or darker, warmer or cooler, as weathered stone and render are.
  vec2 seed = floor(vSpot.xz / 41.0);
  float tone = hash2(seed * 3.1);
  diffuseColor.rgb *= (0.86 + 0.16 * tone) * vec3(1.0 + 0.03 * (tone - 0.5), 1.0, 1.0 - 0.04 * (tone - 0.5));
  bool walls = (kind > 0.5 && kind < 4.5) || (kind > 7.5 && kind < 8.5);
  if (kind > 9.5) {
    // A brick viaduct: arches along it where it stands tall enough, their openings in shadow.
    float wallH = kind - 10.0;
    float x = (fract(uv.x / 9.0) - 0.5) * 9.0;
    float spring = wallH - 3.6;
    float open = step(abs(x), 3.1) * step(uv.y, spring) + step(spring, uv.y) * step(x * x + (uv.y - spring) * (uv.y - spring), 3.1 * 3.1);
    open *= step(6.5, wallH);
    diffuseColor.rgb *= 0.88 + 0.2 * texture2D(uNoise, uv / 1.5).r;
    diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.13, 0.11, 0.1), open * 0.9);
    diffuseColor.rgb *= mix(0.82, 1.0, smoothstep(0.0, 6.0, uv.y));
  }
  if (kind > 2.5 && kind < 3.5) {
    // A curtain wall: glass panels between mullions, an opaque band at each floor's slab,
    // each panel catching the sky a little differently.
    vec2 cell = floor(uv / vec2(1.5, 3.8));
    vec2 fw = fwidth(uv) + 1e-4;
    // Panel to panel variation, fading out where panels are a few pixels (it would sparkle).
    float h = mix(0.5, hash2(cell + seed * 7.0), 1.0 - smoothstep(0.08, 0.25, fw.x / 1.5));
    float mullion = 1.0 - smoothstep(0.05 - fw.x, 0.05 + fw.x, abs(fract(uv.x / 1.5) - 0.5) * 1.5 - 0.7);
    float slab = 1.0 - smoothstep(0.0, fw.y * 2.0 + 0.02, abs(fract(uv.y / 3.8) - 0.1) * 3.8 - 0.35);
    float fade = 1.0 - smoothstep(0.2, 0.6, fw.x / 1.5);
    vec3 pane = mix(vec3(0.17, 0.25, 0.33), vec3(0.33, 0.42, 0.50), h) * (0.9 + 0.2 * grain);
    vec3 frame = diffuseColor.rgb * 0.75;
    vec3 col = mix(pane, frame, max(mullion * 0.6, slab * 0.85) * fade);
    diffuseColor.rgb = mix(diffuseColor.rgb, col, 0.92);
    vFacadeGlass = 0.95 - 0.5 * slab * fade;
  } else if (walls) {
    // Windows on every floor; offices have shop fronts at street level.
    vec2 period = kind < 1.5 ? vec2(3.2, 2.9) : kind < 2.5 ? vec2(3.0, 3.6) : vec2(6.0, 4.5);
    vec2 size = kind < 1.5 ? vec2(1.1, 1.5) : kind < 2.5 ? vec2(2.2, 2.1) : vec2(2.2, 1.0);
    // Each building its own rhythm: storeys a little taller or lower, and offices with
    // punched windows, ribbon glazing or tall bays; homes with wider or narrower sashes.
    float style = hash2(seed * 5.3 + 1.7);
    float storey = 0.92 + 0.18 * hash2(seed * 2.9 + 0.3);
    period.y *= storey;
    size.y *= storey;
    if (kind > 1.5 && kind < 2.5) {
      if (style > 0.66) { period.x = 3.0; size = vec2(2.94, 1.6 * storey); }        // ribbons
      else if (style > 0.4) { period.x = 1.8; size = vec2(1.15, 2.5 * storey); }    // tall bays
    } else if (kind < 1.5 && style > 0.5) {
      period.x = 2.6; size.x = 0.95;
    }
    if (kind > 7.5) { period = vec2(4.2, 5.0); size = vec2(1.4, 2.6); } // tall stone windows
    float base = kind < 2.5 ? 4.4 : 1.2; // the ground floor
    float w = boxes(uv - vec2(0.0, 0.9), period, size) * step(base, uv.y);
    if (kind > 0.5 && kind < 2.5) {
      float shop = boxes(uv - vec2(0.0, 2.2), vec2(6.0, 8.0), vec2(5.0, 3.0)) * (1.0 - step(4.2, uv.y));
      w = max(w, shop);
    }
    vec2 cell = floor((uv - vec2(0.0, 0.9)) / period);
    float h = hash2(cell + seed * 7.0);
    vec3 pane = mix(vec3(0.07, 0.09, 0.12), vec3(0.20, 0.25, 0.31), h * h); // dark, or catching the sky
    if (hash2(cell.yx + seed) > 0.93) pane = vec3(0.40, 0.38, 0.34); // blinds drawn
    // The window's recess: the head of the opening in shadow.
    float inWin = fract((uv.y - 0.9) / period.y + 0.5) * period.y - (period.y - size.y) * 0.5;
    pane *= mix(1.0, 0.6, smoothstep(size.y * 0.75, size.y, inWin));
    // A lintel over each window on homes and stone.
    float lintel = (kind < 1.5 || kind > 7.5) ? boxes(uv - vec2(0.0, 0.9 + size.y * 0.5 + 0.18), period, vec2(size.x + 0.3, 0.22)) : 0.0;
    vec3 wall = diffuseColor.rgb * (0.92 + 0.16 * grain) * (1.0 + 0.12 * lintel);
    diffuseColor.rgb = mix(wall, pane, w * 0.85);
    vFacadeGlass = w * 0.55;
  } else if (kind > 4.5 && kind < 5.5) {
    // Roof tiles: courses up the slope, each tile a little different.
    float course = fract(uv.y / 0.32);
    float tile = texture2D(uNoise, vec2(floor(uv.x / 0.28), floor(uv.y / 0.32)) / 512.0).r;
    float edge = smoothstep(0.0, 0.18, course) * (1.0 - 0.35 * smoothstep(0.82, 1.0, course));
    diffuseColor.rgb *= (0.78 + 0.22 * edge) * (0.9 + 0.2 * tile);
  } else if (kind > 5.5 && kind < 6.5) {
    // Gravel and felt, stained in patches.
    float stain = texture2D(uNoise, uv / 37.0).r;
    diffuseColor.rgb *= (0.74 + 0.22 * grain) * (0.85 + 0.3 * stain);
  } else if (kind > 6.5) {
    diffuseColor.rgb *= 0.88 + 0.2 * texture2D(uNoise, uv / 2.0).r; // stone
  }
  // The street's shade: walls darken towards the ground.
  if (walls || (kind > 6.5 && kind < 7.5)) diffuseColor.rgb *= mix(0.8, 1.0, smoothstep(0.0, 9.0, uv.y));
}`,
      )
      .replace(
        "#include <roughnessmap_fragment>",
        "#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, 0.12, vFacadeGlass);",
      )
      .replace(
        "#include <metalnessmap_fragment>",
        "#include <metalnessmap_fragment>\nmetalnessFactor = mix(metalnessFactor, 0.55, vFacadeGlass);",
      );
  };
  return mat;
}

/** Road surfaces (``aRoad``: along, across, half width, kind): tarmac grain, a dashed centre
 * line on roads, wheel ruts on tracks, soft edges, and a fade into the painted ground beyond
 * a kilometre or so (where a few metres of road are less than a pixel). */
export function roadMaterial(): THREE.MeshLambertMaterial {
  const d = detailTextures();
  const mat = new THREE.MeshLambertMaterial({
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -3,
    polygonOffsetUnits: -3,
  });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uPaved = { value: d.paved };
    sh.uniforms.uGrass = { value: d.grass };
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nattribute vec4 aRoad;\nvarying vec4 vRoad;\nvarying vec3 vWorld;")
      .replace(
        "#include <begin_vertex>",
        "#include <begin_vertex>\nvRoad = aRoad;\nvWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;",
      );
    sh.fragmentShader = sh.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec4 vRoad;
varying vec3 vWorld;
uniform sampler2D uPaved, uGrass;`,
      )
      .replace(
        "#include <color_fragment>",
        `#include <color_fragment>
{
  float along = vRoad.x, across = vRoad.y, hw = vRoad.z, kind = floor(vRoad.w + 0.5);
  vec2 w = vec2(vWorld.x, -vWorld.z);
  vec3 grain = 0.6 * texture2D(uPaved, w / 11.0).rgb + 0.4 * texture2D(uPaved, w / 30.0).rgb;
  diffuseColor.rgb *= 1.0 + 0.9 * (grain - 0.5);
  float fw = fwidth(across) + 1e-3;
  if (kind > 0.5 && kind < 1.5) {
    float dash = step(fract(along / 9.0), 0.45) * (1.0 - smoothstep(0.07, 0.07 + fw, abs(across)));
    float edge = 1.0 - smoothstep(0.06, 0.06 + fw, abs(abs(across) - (hw - 0.35)));
    diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.78), max(dash, edge) * 0.85);
  } else if (kind > 2.5) {
    // A railway: two rails on sleepers in ballast.
    float sleeper = (1.0 - smoothstep(0.1, 0.1 + fwidth(along), abs(fract(along / 0.65) - 0.5) * 0.65 - 0.05))
      * (1.0 - smoothstep(1.2, 1.3, abs(across)));
    float rail = 1.0 - smoothstep(0.04, 0.04 + fw, abs(abs(across) - 0.72));
    float fade = 1.0 - smoothstep(0.15, 0.5, fwidth(along) / 0.65);
    diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb * 0.62, sleeper * fade);
    diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.55, 0.57, 0.6), rail * 0.9);
  } else if (kind > 1.5) {
    // A track: two wheel ruts with grass between them and along the edges.
    float rut = 1.0 - smoothstep(0.35, 0.6, abs(abs(across) - 0.85));
    float g = texture2D(uGrass, w / 4.0).r;
    diffuseColor.rgb = mix(vec3(0.20, 0.30, 0.10) * (0.8 + 0.4 * g), diffuseColor.rgb, rut);
  }
  float dist = length(vWorld - cameraPosition);
  diffuseColor.a *= (1.0 - smoothstep(hw - 0.5, hw, abs(across))) * (1.0 - smoothstep(700.0, 1500.0, dist));
}`,
      );
  };
  return mat;
}

/** Water: dark, smooth and rippled, reflecting the sky (``scene.environment``). */
export function waterMaterial(): THREE.MeshStandardMaterial {
  const d = detailTextures();
  const normal = d.waterNormal.clone();
  normal.needsUpdate = true;
  normal.repeat.set(1 / 9, 1 / 9);
  const mat = new THREE.MeshStandardMaterial({
    color: "#2b4c55", // a river's grey-green, under the sky it reflects
    roughness: 0.06,
    metalness: 0.0,
    normalMap: normal,
    normalScale: new THREE.Vector2(0.35, 0.35),
    envMapIntensity: 1.6,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  });
  mat.userData.flow = normal; // scrolled each frame
  return mat;
}
