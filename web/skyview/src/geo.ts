import { Cartesian3, Cartographic, Matrix4, Transforms } from "cesium";

export interface Origin {
  lat_deg: number;
  lon_deg: number;
  h_ellipsoid_m: number;
}

/**
 * Local NED (site origin) → Earth-fixed position. Horizontal: the local tangent plane at the
 * origin. Height: WGS84 ellipsoidal h = origin.h + alt_agl (flat-ground M1 site, spec §11).
 */
export class SiteFrame {
  private readonly enu: Matrix4;
  private readonly scratchLocal = new Cartesian3();
  private readonly scratchCarto = new Cartographic();

  constructor(readonly origin: Origin) {
    this.enu = Transforms.eastNorthUpToFixedFrame(
      Cartesian3.fromDegrees(origin.lon_deg, origin.lat_deg, origin.h_ellipsoid_m),
    );
  }

  toCartesian(n: number, e: number, altAgl: number, result = new Cartesian3()): Cartesian3 {
    this.scratchLocal.x = e;
    this.scratchLocal.y = n;
    this.scratchLocal.z = 0;
    Matrix4.multiplyByPoint(this.enu, this.scratchLocal, result);
    const c = Cartographic.fromCartesian(result, undefined, this.scratchCarto);
    return Cartesian3.fromRadians(
      c.longitude,
      c.latitude,
      this.origin.h_ellipsoid_m + altAgl,
      undefined,
      result,
    );
  }

  /** Ground point (alt_agl = 0) as degrees, for polygons clamped to the ground. */
  toDegrees(n: number, e: number): [number, number] {
    const p = this.toCartesian(n, e, 0);
    const c = Cartographic.fromCartesian(p);
    return [(c.longitude * 180) / Math.PI, (c.latitude * 180) / Math.PI];
  }
}
