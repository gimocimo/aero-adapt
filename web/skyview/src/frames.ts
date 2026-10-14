// Frame conventions for the viewer (ADR-001). The source of truth is aeroadapt.core
// (Python); src/frames.test.ts checks this file against goldens exported from Python.
//
// Simulator: body FRD, world NED, attitude quaternion scalar-first body→NED (Hamilton).
// CesiumJS: model frame FLU (forward, left, up) after its glTF Y-up/Z-forward correction,
// local world frame ENU.

export type Quat = readonly [number, number, number, number];
export type Vec3 = readonly [number, number, number];
/** Row-major 3x3 matrix. */
export type Mat3 = readonly [Vec3, Vec3, Vec3];

const DEG = Math.PI / 180;

/** Quaternion from ZYX Euler angles (roll φ, pitch θ, yaw ψ), in radians. */
export function eulerToQuat(phi: number, theta: number, psi: number): Quat {
  const [cp, sp] = [Math.cos(phi / 2), Math.sin(phi / 2)];
  const [ct, st] = [Math.cos(theta / 2), Math.sin(theta / 2)];
  const [cy, sy] = [Math.cos(psi / 2), Math.sin(psi / 2)];
  return [
    cy * ct * cp + sy * st * sp,
    cy * ct * sp - sy * st * cp,
    cy * st * cp + sy * ct * sp,
    sy * ct * cp - cy * st * sp,
  ];
}

export function eulerDegToQuat(rollDeg: number, pitchDeg: number, headingDeg: number): Quat {
  return eulerToQuat(rollDeg * DEG, pitchDeg * DEG, headingDeg * DEG);
}

/** Direction cosine matrix R_ned_body (body FRD → NED). */
export function quatToDcm([e0, e1, e2, e3]: Quat): Mat3 {
  return [
    [e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2, 2 * (e1 * e2 - e3 * e0), 2 * (e1 * e3 + e2 * e0)],
    [2 * (e1 * e2 + e3 * e0), e0 ** 2 - e1 ** 2 + e2 ** 2 - e3 ** 2, 2 * (e2 * e3 - e1 * e0)],
    [2 * (e1 * e3 - e2 * e0), 2 * (e2 * e3 + e1 * e0), e0 ** 2 - e1 ** 2 - e2 ** 2 + e3 ** 2],
  ];
}

export function matMul(a: Mat3, b: Mat3): Mat3 {
  const row = (i: 0 | 1 | 2): Vec3 => [
    a[i][0] * b[0][0] + a[i][1] * b[1][0] + a[i][2] * b[2][0],
    a[i][0] * b[0][1] + a[i][1] * b[1][1] + a[i][2] * b[2][1],
    a[i][0] * b[0][2] + a[i][1] * b[1][2] + a[i][2] * b[2][2],
  ];
  return [row(0), row(1), row(2)];
}

export function matVec(a: Mat3, v: Vec3): Vec3 {
  return [
    a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
    a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
    a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
  ];
}

/** NED vectors → ENU. */
export const R_ENU_NED: Mat3 = [
  [0, 1, 0],
  [1, 0, 0],
  [0, 0, -1],
];

/** FLU vectors → FRD. */
export const R_FRD_FLU: Mat3 = [
  [1, 0, 0],
  [0, -1, 0],
  [0, 0, -1],
];

/** Rotation taking Cesium model-frame (FLU) vectors into the local ENU frame. */
export function rEnuFromFlu(q: Quat): Mat3 {
  return matMul(matMul(R_ENU_NED, quatToDcm(q)), R_FRD_FLU);
}
