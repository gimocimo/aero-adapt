// Simulator attitudes in the three.js world (ADR-001 conventions, see src/frames.ts).
//
// World: x = east, y = up, z = south. Aircraft models are glTF: +Z forward, +Y up, +X left.
// Body FRD vector b → model: (−b_y, −b_z, b_x); NED vector → world: (e, −d, −n).
import { matMul, type Mat3, quatToDcm, type Quat } from "../frames";

/** NED vectors → three.js world. */
export const R_WORLD_NED: Mat3 = [
  [0, 1, 0],
  [0, 0, -1],
  [-1, 0, 0],
];

/** glTF model vectors → body FRD. */
export const R_FRD_MODEL: Mat3 = [
  [0, 0, 1],
  [-1, 0, 0],
  [0, -1, 0],
];

/** Rotation taking model-frame vectors into the three.js world for attitude ``q`` (body FRD
 * → NED, scalar first). */
export function rWorldFromModel(q: Quat): Mat3 {
  return matMul(matMul(R_WORLD_NED, quatToDcm(q)), R_FRD_MODEL);
}

/** Site NE metres and height above the datum → three.js world coordinates. */
export function toWorld(n: number, e: number, h: number): [number, number, number] {
  return [e, h, -n];
}
