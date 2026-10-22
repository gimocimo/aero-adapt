// Replay bundle types (ADR-005). The Python side (aeroadapt.telemetry) is the source of truth.

export interface SafeZone {
  id: string;
  polygon_ne_m: [number, number][];
  surface: string;
  quality: number;
}

export interface Manifest {
  schema_version: number;
  /** A bundle-relative glTF of the airframe flown (fleet airframes; set by export-viz). */
  viz_model?: string;
  scenario: {
    id: string;
    airframe?: string;
    fleet?: string | null;
    description: string;
    faults: { type: string; trigger: { at_time_s: number } }[];
    environment: { wind: { from_deg: number; speed_mps: number }; turbulence: { intensity: string } };
    stack: { contingency: string };
  };
  site: {
    id: string;
    name: string;
    origin: { lat_deg: number; lon_deg: number; h_ellipsoid_m: number };
    /** A world site's world (docs/07); absent or null on a flat site. */
    world?: string | null;
    safe_zones: SafeZone[];
  };
  mission: { waypoints_ne_alt_m: [number, number, number][]; on_complete: string };
  rates: { control_hz: number; physics_hz: number; viz_hz: number };
  n_frames: number;
  duration_s: number;
  viz_columns: string[];
  enums: { mode: string[]; phase: string[]; fault_bits: Record<string, number> };
}

export interface ReplayEvent {
  time_us: number;
  type: string;
  payload: Record<string, unknown>;
}

export type Column = Float32Array | Int32Array;

export interface Replay {
  id: string;
  manifest: Manifest;
  events: ReplayEvent[];
  /** Sample times (s), strictly increasing. */
  t: Float32Array;
  cols: Record<string, Column>;
  n: number;
  duration: number;
}
