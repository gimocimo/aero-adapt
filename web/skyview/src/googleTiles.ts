import { type Cesium3DTileset, createGooglePhotorealistic3DTileset } from "cesium";

// Without a Google key, Cesium loads the tiles through Cesium ion (asset 2275207) and shares
// one cached IonResource across calls. Two *concurrent* calls therefore throw "The Resource is
// already being fetched". React StrictMode mounts the viewer twice in development, which
// triggers exactly that. Sequential calls are fine, so the calls are queued.
let queue: Promise<unknown> = Promise.resolve();

export function loadGoogleTiles(): Promise<Cesium3DTileset> {
  const next = queue.then(() =>
    // Google 3D Tiles may only be paired with Google's geocoder; the viewer uses none.
    createGooglePhotorealistic3DTileset({ onlyUsingWithGoogleGeocoder: true }),
  );
  queue = next.catch(() => undefined);
  return next;
}
