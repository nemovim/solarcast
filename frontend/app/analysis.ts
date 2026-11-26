import type { FeatureCollection, Geometry } from 'geojson';

export type Summary = { buildingId: string; name: string; availableAreaM2: number | null; panelAreaM2: number | null; capacityKw: number | null; annualEnergyKwh: number | null; heightSource: string };
export type AnalysisResult = {
  buildings: FeatureCollection<Geometry>;
  existing_panels: FeatureCollection<Geometry>;
  available_roofs: FeatureCollection<Geometry>;
  summaries: Summary[];
  warnings: string[];
  totals: { availableAreaM2: number | null; panelAreaM2: number | null; capacityKw: number | null; annualEnergyKwh: number | null };
  models: { roof: string | null; panel: string };
};
