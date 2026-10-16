// The Cesium viewer every Skyview mode uses: Google Photorealistic 3D Tiles when an ion token is
// configured (and ?tiles=0 is not set), else ion terrain with a grid, else the bare ellipsoid.
import { CesiumTerrainProvider, GridImageryProvider, ImageryLayer, Ion, Viewer } from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import { loadGoogleTiles } from "./googleTiles";

const TOKEN = import.meta.env.VITE_CESIUM_ION_TOKEN?.trim();

export function createViewer(
  container: HTMLElement,
  tiles: boolean,
  onStatus: (status: string) => void,
): { viewer: Viewer; destroy: () => void } {
  const useTiles = Boolean(TOKEN) && tiles;
  if (TOKEN) Ion.defaultAccessToken = TOKEN;
  const viewer = new Viewer(container, {
    timeline: false,
    animation: false,
    baseLayerPicker: false,
    geocoder: false,
    homeButton: false,
    sceneModePicker: false,
    navigationHelpButton: false,
    fullscreenButton: false,
    infoBox: false,
    selectionIndicator: false,
    baseLayer: useTiles ? false : new ImageryLayer(new GridImageryProvider({})),
  });
  if (import.meta.env.DEV) (window as unknown as { __skyview?: Viewer }).__skyview = viewer;
  let destroyed = false;
  if (useTiles) {
    viewer.scene.globe.show = false;
    loadGoogleTiles()
      .then((tileset) => {
        if (destroyed) return tileset.destroy();
        viewer.scene.primitives.add(tileset);
        if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__tileset = tileset;
        onStatus("Google Photorealistic 3D Tiles via Cesium ion");
      })
      .catch((e: unknown) => onStatus(`3D tiles failed: ${String(e)}`));
  } else if (TOKEN) {
    CesiumTerrainProvider.fromIonAssetId(1)
      .then((tp) => !destroyed && (viewer.terrainProvider = tp))
      .catch(() => undefined);
    onStatus("Tiles off (?tiles=0): terrain + grid");
  } else {
    onStatus("No Cesium ion token: ellipsoid + grid fallback");
  }
  return {
    viewer,
    destroy: () => {
      destroyed = true;
      viewer.destroy();
    },
  };
}
