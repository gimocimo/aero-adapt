// The cinematic look (docs/07 §15): a sky with the sun in it, sunlight with soft
// shadows over the whole map, haze with distance, the sky's light on every surface (so water
// and glass reflect it) and filmic colour grading. For video; ``?quality=low`` turns it off.
import * as THREE from "three";

/** Where the sun is: from the west-south-west, a late-afternoon height (as the sketches). */
export const SUN_AZIMUTH_DEG = 250;
export const SUN_ELEVATION_DEG = 32;
export const HAZE = new THREE.Color("#bfd2e4");
const SHADOW_MAP_MAX = 8192; // texels along the shadow map's longer side, at most
const SHADOW_TEXEL_BIAS = 1.5; // the normal bias, in shadow-map texels

export function sunDirection(): THREE.Vector3 {
  const az = THREE.MathUtils.degToRad(SUN_AZIMUTH_DEG);
  const el = THREE.MathUtils.degToRad(SUN_ELEVATION_DEG);
  // (n, e) → (x = e, z = -n); towards the sun.
  return new THREE.Vector3(Math.cos(el) * Math.sin(az), Math.sin(el), -Math.cos(el) * Math.cos(az)).normalize();
}

const ZENITH = new THREE.Color("#2f6fc0");
const SUN_GLOW = new THREE.Color("#fff3d6");

/** The sky: deep blue overhead fading to the haze at the horizon, the sun's disc and its
 * glow. (A physical sky model is too bright at this sun height for the exposure the
 * ground needs, and filmic grading turns it white.) */
function skyDome(radius: number): THREE.Mesh {
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uGain: { value: 1.55 }, // through the same grading as the ground, so the horizon meets the haze
      uZenith: { value: ZENITH },
      uHorizon: { value: HAZE },
      uSun: { value: sunDirection() },
      uGlow: { value: SUN_GLOW },
    },
    vertexShader: `varying vec3 vDir;
void main() {
  vDir = normalize((modelMatrix * vec4(position, 1.0)).xyz - cameraPosition);
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
  gl_Position.z = gl_Position.w; // on the far plane
}`,
    fragmentShader: `uniform vec3 uZenith, uHorizon, uSun, uGlow;
uniform float uGain;
varying vec3 vDir;
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = fract(sin(dot(i, vec2(127.1, 311.7))) * 43758.5453);
  float b = fract(sin(dot(i + vec2(1.0, 0.0), vec2(127.1, 311.7))) * 43758.5453);
  float c = fract(sin(dot(i + vec2(0.0, 1.0), vec2(127.1, 311.7))) * 43758.5453);
  float d = fract(sin(dot(i + vec2(1.0, 1.0), vec2(127.1, 311.7))) * 43758.5453);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int k = 0; k < 5; k++) { v += a * vnoise(p); p = p * 2.03 + vec2(17.0, 9.0); a *= 0.5; }
  return v;
}
void main() {
  vec3 d = normalize(vDir);
  float up = clamp(d.y, 0.0, 1.0);
  vec3 col = mix(uHorizon, uZenith, pow(up, 0.5));
  col = mix(col, uHorizon * 0.92, smoothstep(0.0, -0.08, d.y)); // below the horizon: haze
  float s = max(dot(d, normalize(uSun)), 0.0);
  col += uGlow * (0.22 * pow(s, 10.0) + 0.5 * pow(s, 90.0)) + uGlow * smoothstep(0.9995, 0.99975, s) * 2.5;
  // Fair-weather cumulus on a layer 1.5 km up: lit from the sun, greyer underneath, thinning
  // into the haze towards the horizon.
  if (d.y > 0.0) {
    vec2 p = d.xz / max(d.y, 0.02) * 1.5;
    float n = fbm(p * 0.9 + vec2(3.0, 1.0));
    float cover = smoothstep(0.52, 0.72, n) * smoothstep(0.02, 0.18, d.y);
    float lit = 0.75 + 0.35 * smoothstep(0.45, 0.85, fbm(p * 0.9 + normalize(uSun).xz * 0.08 + vec2(3.0, 1.0)));
    vec3 cloud = mix(vec3(0.78, 0.80, 0.84), vec3(1.0, 0.98, 0.95), clamp(lit - 0.6, 0.0, 1.0)) * lit;
    col = mix(col, cloud, cover * 0.92);
  }
  gl_FragColor = vec4(col * uGain, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 48, 24), mat);
  mesh.frustumCulled = false;
  mesh.renderOrder = -1;
  return mesh;
}

export class Look {
  readonly sun: THREE.DirectionalLight;
  readonly hemi: THREE.HemisphereLight;
  readonly sky: THREE.Mesh;
  private env: THREE.Texture | null = null;

  constructor(
    private renderer: THREE.WebGLRenderer,
    private scene: THREE.Scene,
  ) {
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.62;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap; // the soft kind cost 1.5 ms a frame more

    this.sky = skyDome(24000);
    scene.add(this.sky);

    // The sky's light, for image-based lighting (water and glass reflect it).
    const pmrem = new THREE.PMREMGenerator(renderer);
    const skyScene = new THREE.Scene();
    skyScene.add(skyDome(100));
    this.env = pmrem.fromScene(skyScene, 0, 0.1, 500).texture;
    pmrem.dispose();
    scene.environment = this.env;
    scene.environmentIntensity = 0.6;

    this.hemi = new THREE.HemisphereLight("#d5e4f3", "#5b6b48", 1.0);
    this.sun = new THREE.DirectionalLight("#fff0da", 4.6);
    this.sun.castShadow = true;
    const s = this.sun.shadow;
    s.radius = 2.5;
    scene.add(this.hemi, this.sun, this.sun.target);
    // Shadows are cast once a world is loaded (``shadowWorld``), not every frame.
    renderer.shadowMap.autoUpdate = false;
  }

  /** Cast the sun's shadows over the whole of ``box`` (a world's ground and what stands on it),
   * once: one shadow map, fitted to the box as the sun sees it, drawn on the next frame. What
   * moves (the aircraft, the turbines' blades) casts none, so nothing appears or changes as
   * the camera travels. On the test map a texel is 0.78 m. */
  shadowWorld(box: THREE.Box3): void {
    const centre = box.getCenter(new THREE.Vector3());
    const reach = box.getSize(new THREE.Vector3()).length();
    this.sun.target.position.copy(centre);
    this.sun.position.copy(centre).addScaledVector(sunDirection(), reach);
    this.sun.target.updateMatrixWorld();
    this.sun.updateMatrixWorld();
    // The tightest frame round the box's corners, in the light's view.
    const cam = this.sun.shadow.camera;
    cam.position.copy(this.sun.position);
    cam.lookAt(centre);
    cam.updateMatrixWorld();
    const lo = new THREE.Vector3(Infinity, Infinity, Infinity);
    const hi = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    for (const x of [box.min.x, box.max.x])
      for (const y of [box.min.y, box.max.y])
        for (const z of [box.min.z, box.max.z]) {
          const p = new THREE.Vector3(x, y, z).applyMatrix4(cam.matrixWorldInverse);
          lo.min(p);
          hi.max(p);
        }
    [cam.left, cam.right, cam.bottom, cam.top] = [lo.x, hi.x, lo.y, hi.y];
    cam.near = Math.max(1, -hi.z - 10);
    cam.far = -lo.z + 10;
    cam.updateProjectionMatrix();
    const [w, h] = [hi.x - lo.x, hi.y - lo.y];
    const most = Math.min(SHADOW_MAP_MAX, this.renderer.capabilities.maxTextureSize);
    const sx = w >= h ? most : Math.ceil((most * w) / h);
    const sy = w >= h ? Math.ceil((most * h) / w) : most;
    const s = this.sun.shadow;
    s.mapSize.set(sx, sy);
    s.map?.dispose();
    s.map = null; // made again at the new size
    const texel = Math.max(w / sx, h / sy);
    s.normalBias = SHADOW_TEXEL_BIAS * texel;
    s.bias = -0.5 / (cam.far - cam.near); // half a metre, in the map's depth
    this.renderer.shadowMap.needsUpdate = true;
  }

  dispose(): void {
    this.scene.remove(this.sky, this.hemi, this.sun, this.sun.target);
    this.sky.geometry.dispose();
    (this.sky.material as THREE.Material).dispose();
    this.env?.dispose();
    this.scene.environment = null;
    this.renderer.shadowMap.enabled = false;
  }
}
