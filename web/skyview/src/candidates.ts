// UK hero-site candidates for the week-1 tile survey (docs/04 §11.1). Coordinates are
// approximate starting points; the survey refines them. If you know the club field you
// will fly from in months 9-12, add it here: the simulated site then becomes a digital twin.

export interface SiteCandidate {
  id: string;
  name: string;
  lat: number;
  lon: number;
  note: string;
}

export const CANDIDATES: readonly SiteCandidate[] = [
  {
    id: "south_downs",
    name: "South Downs — Devil's Dyke (near Brighton)",
    lat: 50.8853,
    lon: -0.2133,
    note: "Chalk ridges, fields, town edge, coastline",
  },
  {
    id: "cambridge_west",
    name: "Cambridge west fringe — Coton",
    lat: 52.2075,
    lon: 0.06,
    note: "Flat farmland, villages, M11; check nearby aerodrome airspace",
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
