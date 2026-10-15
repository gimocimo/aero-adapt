// Trail colour categories (flight mode / contingency phase), shared by the scene and the HUD.

export interface Category {
  key: string;
  label: string;
  css: string;
}

export const CATEGORIES: Category[] = [
  { key: "nominal", label: "Mission", css: "#4cc2ff" },
  { key: "circle", label: "Hold / circle", css: "#ff9f43" },
  { key: "pattern", label: "Downwind / base", css: "#ffd84c" },
  { key: "final", label: "Final / flare", css: "#7dff7a" },
  { key: "straight", label: "Straight in", css: "#ff5c5c" },
  { key: "rtl", label: "RTL", css: "#c792ea" },
];

/** Category index for a phase name (contingency) and mode name. */
export function categoryOf(phase: string, mode: string): number {
  switch (phase) {
    case "CIRCLE":
      return 1;
    case "DOWNWIND":
    case "BASE":
      return 2;
    case "FINAL":
    case "FLARE":
      return 3;
    case "STRAIGHT_IN":
      return 4;
    default:
      return mode === "RTL" ? 5 : 0;
  }
}
