import { Cartesian3, Matrix3, Matrix4, Quaternion, Transforms } from "cesium";
import { rEnuFromFlu, type Quat } from "./frames";

/**
 * Cesium entity orientation (model FLU frame → Earth-fixed frame) for an aircraft at
 * `position` with simulator attitude `q` (body FRD → NED, scalar-first). This is the single
 * place where simulator attitudes meet CesiumJS (ADR-001); calibrate it visually with the
 * nav-light model after any change.
 */
export function orientationFromNedQuat(position: Cartesian3, q: Quat): Quaternion {
  const rEcefEnu = Matrix4.getMatrix3(Transforms.eastNorthUpToFixedFrame(position), new Matrix3());
  const rEnuFlu = Matrix3.fromRowMajorArray(rEnuFromFlu(q).flat());
  return Quaternion.fromRotationMatrix(Matrix3.multiply(rEcefEnu, rEnuFlu, new Matrix3()));
}
