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
    });
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vWorld;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;");
    sh.fragmentShader = sh.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vWorld;
uniform sampler2D uClass, uGrass, uCrop, uSoil, uPaved, uForest;`,
      )
      .replace(
        "#include <map_fragment>",
        `#include <map_fragment>
{
  float cls = floor(texture2D(uClass, vMapUv).r * 255.0 / ${CLASS_STEP.toFixed(1)} + 0.5);
  vec2 w = vec2(vWorld.x, -vWorld.z);
  float dist = length(vWorld - cameraPosition);
  float near = 1.0 - smoothstep(180.0, 1600.0, dist);
  float g = 0.5;
  if (cls < 0.5) g = 0.6 * texture2D(uGrass, w / 5.0).r + 0.4 * texture2D(uGrass, w / 23.0).r;
  else if (cls < 1.5) g = 0.6 * texture2D(uCrop, w / 6.0).r + 0.4 * texture2D(uGrass, w / 29.0).r;
  else if (cls < 2.5) g = 0.65 * texture2D(uSoil, w / 5.0).r + 0.35 * texture2D(uSoil, w / 27.0).r;
  else if (cls < 3.5) g = 0.7 * texture2D(uPaved, w / 4.0).r + 0.3 * texture2D(uPaved, w / 31.0).r;
  else if (cls < 4.5) g = 0.6 * texture2D(uForest, w / 4.0).r + 0.4 * texture2D(uForest, w / 19.0).r;
  float strength = cls > 4.5 ? 0.0 : (cls > 2.5 && cls < 3.5 ? 1.0 : 1.6);
  diffuseColor.rgb *= 1.0 + near * strength * (g - 0.5);
  // Patches a few tens of metres across (thinner and lusher grass, wetter soil), out to a few
  // kilometres, and broad mottling everywhere: no field is one flat colour.
  float mid = 0.6 * texture2D(uGrass, w / 70.0).r + 0.4 * texture2D(uSoil, w / 150.0).r;
  float midNear = 1.0 - smoothstep(400.0, 3500.0, dist);
  diffuseColor.rgb *= 1.0 + (cls < 2.5 ? 1.3 : 0.5) * midNear * (mid - 0.5);
  float broad = texture2D(uGrass, w / 260.0).r * 0.6 + texture2D(uSoil, w / 900.0).r * 0.4;
  diffuseColor.rgb *= 1.0 + 0.6 * (broad - 0.5);
}`,
      );
  };
  return mat;
}

/** Facade and roof kinds, per vertex (``aFacade.z``). */
export const FACADE = { none: 0, homes: 1, offices: 2, glass: 3, industry: 4, tiles: 5, flat: 6, stone: 7 } as const;

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
    sh.uniforms.uNoise = { value: d.paved };
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nattribute vec3 aFacade;\nvarying vec3 vFacade;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvFacade = aFacade;");
    sh.fragmentShader = sh.fragmentShader
      .replace(
        "#include <common>",
        `#include <common>
varying vec3 vFacade;
uniform sampler2D uNoise;
float vFacadeGlass = 0.0;
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
  if (kind > 0.5 && kind < 4.5) {
    // Walls: windows on every floor above the ground floor (shop fronts below on offices).
    vec2 period = kind < 1.5 ? vec2(3.2, 2.9) : kind < 2.5 ? vec2(3.0, 3.6) : vec2(1.5, 3.8);
    vec2 size = kind < 1.5 ? vec2(1.2, 1.4) : kind < 2.5 ? vec2(1.9, 2.0) : vec2(1.38, 3.55);
    if (kind > 3.5) { period = vec2(6.0, 4.5); size = vec2(2.2, 1.0); }
    float w = boxes(uv - vec2(0.0, 0.9), period, size) * step(1.2, uv.y);
    vec3 pane = kind > 2.5 && kind < 3.5 ? vec3(0.22, 0.32, 0.42) : vec3(0.10, 0.13, 0.17);
    diffuseColor.rgb = mix(diffuseColor.rgb * (0.92 + 0.16 * grain), pane, w * (kind > 2.5 && kind < 3.5 ? 0.9 : 0.78));
    vFacadeGlass = w * (kind > 2.5 && kind < 3.5 ? 1.0 : 0.55);
  } else if (kind > 4.5 && kind < 5.5) {
    // Roof tiles: courses up the slope, each tile a little different.
    float course = fract(uv.y / 0.32);
    float tile = texture2D(uNoise, vec2(floor(uv.x / 0.28), floor(uv.y / 0.32)) / 512.0).r;
    float edge = smoothstep(0.0, 0.18, course) * (1.0 - 0.35 * smoothstep(0.82, 1.0, course));
    diffuseColor.rgb *= (0.78 + 0.22 * edge) * (0.9 + 0.2 * tile);
  } else if (kind > 5.5 && kind < 6.5) {
    diffuseColor.rgb *= 0.86 + 0.24 * grain; // gravel and felt
  } else if (kind > 6.5) {
    diffuseColor.rgb *= 0.88 + 0.2 * texture2D(uNoise, uv / 2.0).r; // stone
  }
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
  float grain = 0.6 * texture2D(uPaved, w / 3.0).r + 0.4 * texture2D(uPaved, w / 17.0).r;
  diffuseColor.rgb *= 0.82 + 0.36 * grain;
  float fw = fwidth(across) + 1e-3;
  if (kind > 0.5 && kind < 1.5) {
    float dash = step(fract(along / 9.0), 0.45) * (1.0 - smoothstep(0.07, 0.07 + fw, abs(across)));
    float edge = 1.0 - smoothstep(0.06, 0.06 + fw, abs(abs(across) - (hw - 0.35)));
    diffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.78), max(dash, edge) * 0.85);
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
    color: "#25526b",
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
