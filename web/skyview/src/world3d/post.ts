// Post-processing for the cinematic chase view (docs/07 §15): ambient occlusion (the dark of
// a street between tall buildings, under eaves and trees) and multisampled edges, then the
// filmic grading and a lens's vignette. It draws into whatever viewport the renderer has set.
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { GTAOPass } from "three/examples/jsm/postprocessing/GTAOPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { VignetteShader } from "three/examples/jsm/shaders/VignetteShader.js";

export class Post {
  private readonly composer: EffectComposer;
  private readonly ao: GTAOPass;
  private w = 0;
  private h = 0;

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
  ) {
    const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(renderer, target);
    this.composer.addPass(new RenderPass(scene, camera));
    this.ao = new GTAOPass(scene, camera, 1, 1);
    // In metres: occlusion reaches a few metres (a street's walls, a courtyard, a tree's foot).
    this.ao.updateGtaoMaterial({ radius: 5, distanceExponent: 2, thickness: 4, distanceFallOff: 1, scale: 1.3, samples: 16 });
    this.ao.updatePdMaterial({ lumaPhi: 10, depthPhi: 2, normalPhi: 3, radius: 5, rings: 2, samples: 16 });
    this.ao.blendIntensity = 0.85;
    this.composer.addPass(this.ao);
    this.composer.addPass(new OutputPass());
    // A lens's slight fall-off towards the corners.
    const vignette = new ShaderPass(VignetteShader);
    vignette.uniforms.offset!.value = 0.95;
    vignette.uniforms.darkness!.value = 0.65;
    this.composer.addPass(vignette);
  }

  /** Draw the scene through the passes into the renderer's current viewport (w × h CSS px). */
  render(w: number, h: number): void {
    if (w !== this.w || h !== this.h) {
      this.w = w;
      this.h = h;
      this.composer.setPixelRatio(this.renderer.getPixelRatio());
      this.composer.setSize(w, h);
    }
    this.composer.render();
  }

  dispose(): void {
    this.composer.dispose();
    this.ao.dispose();
  }
}
