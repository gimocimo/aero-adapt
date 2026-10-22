// The cinematic look (docs/07 §15): a sky with the sun in it, sunlight with soft
// shadows round the aircraft, haze with distance, the sky's light on every surface (so water
// and glass reflect it) and filmic colour grading. For video; ``?quality=low`` turns it off.
import * as THREE from "three";

/** Where the sun is: from the west-south-west, a late-afternoon height (as the sketches). */
export const SUN_AZIMUTH_DEG = 250;
export const SUN_ELEVATION_DEG = 32;
export const HAZE = new THREE.Color("#bfd2e4");
const SHADOW_HALF_M = 320; // shadows within this of the chase camera's target

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
void main() {
  vec3 d = normalize(vDir);
  float up = clamp(d.y, 0.0, 1.0);
  vec3 col = mix(uHorizon, uZenith, pow(up, 0.5));
  col = mix(col, uHorizon * 0.92, smoothstep(0.0, -0.08, d.y)); // below the horizon: haze
  float s = max(dot(d, normalize(uSun)), 0.0);
  col += uGlow * (0.22 * pow(s, 10.0) + 0.5 * pow(s, 90.0)) + uGlow * smoothstep(0.9995, 0.99975, s) * 2.5;
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
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

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
    s.mapSize.set(4096, 4096);
    s.camera.left = -SHADOW_HALF_M;
    s.camera.right = SHADOW_HALF_M;
    s.camera.top = SHADOW_HALF_M;
    s.camera.bottom = -SHADOW_HALF_M;
    s.camera.near = 1;
    s.camera.far = 4000;
    s.bias = -0.0004;
    s.normalBias = 0.5;
    s.radius = 2.5;
    scene.add(this.hemi, this.sun, this.sun.target);
  }

  /** Keep the sunlit, shadowed box over what the chase camera looks at. */
  follow(target: THREE.Vector3): void {
    const dir = sunDirection();
    // Snap to shadow-map texels so shadows do not shimmer as the box moves.
    const texel = (2 * SHADOW_HALF_M) / this.sun.shadow.mapSize.x;
    const t = new THREE.Vector3(
      Math.round(target.x / texel) * texel,
      Math.round(target.y / texel) * texel,
      Math.round(target.z / texel) * texel,
    );
    this.sun.target.position.copy(t);
    this.sun.position.copy(t).addScaledVector(dir, 2000);
    this.sun.target.updateMatrixWorld();
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
