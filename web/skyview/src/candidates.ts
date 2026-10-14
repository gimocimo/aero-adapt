// UK sites for the viewer. The first entry is the chosen development site, which mirrors
// configs/sites/cambridge_west.yaml (origin and surveyed ground height). The others are the
// week-1 survey candidates (docs/04 §11.1, tools/tile_survey.md), with approximate coordinates.

export interface SiteCandidate {
  id: string;
  name: string;
  lat: number;
  lon: number;
  note: string;
  /** Surveyed WGS84 ellipsoidal ground height from the site config; else terrain is sampled. */
  hEllipsoidM?: number;
}

export const CANDIDATES: readonly SiteCandidate[] = [
  {
    id: "cambridge_west",
    name: "Cambridge west — Coton fields (dev site)",
    lat: 52.2075,
    lon: 0.0747,
    hEllipsoidM: 63.9,
    note: "Chosen dev site: flat arable field between Coton and the M11; West Cambridge campus NE",
  },
  {
    id: "south_downs",
    name: "South Downs — Devil's Dyke (near Brighton)",
    lat: 50.8853,
    lon: -0.2133,
    note: "Chalk ridges, fields, town edge, coastline",
  },
  {
    id: "bath_lansdown",
    name: "Bath north fringe — Lansdown",
    lat: 51.421,
    lon: -2.385,
    note: "Plateau and valleys, city nearby",
  },
  {
    id: "peak_hope_valley",
    name: "Peak District edge — Hope Valley",
    lat: 53.344,
    lon: -1.776,
    note: "Moorland, valleys, reservoirs; rural 3D coverage uncertain",
  },
];
