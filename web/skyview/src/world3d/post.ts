// Post-processing for the cinematic chase view (docs/07 §15): multisampled edges, then the
// filmic grading and a lens's vignette. It draws into whatever viewport the renderer has set.
//
// There was ambient occlusion here too (the dark of a street between tall buildings, under
// eaves and trees). It rendered the scene a second time, for its depth and normals, and cost
// about 4 ms of a frame whatever its resolution or samples: it went for 60 frames a second
// (the owner's choice, 25 Oct 2026).
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { VignetteShader } from "three/examples/jsm/shaders/VignetteShader.js";

export class Post {
  private readonly composer: EffectComposer;
  private w = 0;
  private h = 0;
  private pr = 0;

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
  ) {
    const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(renderer, target);
    this.composer.addPass(new RenderPass(scene, camera));
    this.composer.addPass(new OutputPass());
    // A lens's slight fall-off towards the corners.
    const vignette = new ShaderPass(VignetteShader);
    vignette.uniforms.offset!.value = 0.95;
    vignette.uniforms.darkness!.value = 0.65;
    this.composer.addPass(vignette);
  }

  /** Draw the scene through the passes into the renderer's current viewport (w × h CSS px). */
  render(w: number, h: number): void {
    const pr = this.renderer.getPixelRatio();
    if (w !== this.w || h !== this.h || pr !== this.pr) {
      this.w = w;
      this.h = h;
      this.pr = pr;
      this.composer.setPixelRatio(pr);
      this.composer.setSize(w, h);
    }
    this.composer.render();
  }

  dispose(): void {
    this.composer.dispose();
  }
}
