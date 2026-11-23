import type { Feature, FeatureCollection, Polygon } from 'geojson';
import type { Map as MapLibreMap } from 'maplibre-gl';

export const LAYERS = [
  { id: 'buildings', name: '3D 건물', color: '#768b80' },
  { id: 'shadows', name: '옥상 음영', color: '#997bd9' },
  { id: 'panels', name: '기존 패널', color: '#65c4e4' },
  { id: 'available', name: '설치 가능 영역', color: '#e8ec85' },
] as const;

type LayerId = (typeof LAYERS)[number]['id'];
export type LayerVisibility = Record<LayerId, boolean>;
export const DEFAULT_LAYERS: LayerVisibility = { buildings: true, shadows: true, panels: true, available: true };
type SceneMap = MapLibreMap;

export const DEMO_BUILDING = {
  id: 'solarcast-demo-01', name: '가상 건물 01', longitude: 126.97795, latitude: 37.56632,
  heightM: 31, availableAreaM2: 284, capacityKw: 48, annualEnergyKwh: 58920,
};

const SOURCE = 'solarcast-demo';
const layerKey = (id: LayerId) => `${SOURCE}-${id}`;

// Hand-authored rectangles, not imagery-derived footprints or computed shadows.
function rectangle(kind: LayerId, west: number, south: number, east: number, north: number): Feature<Polygon> {
  const { longitude, latitude, heightM, id } = DEMO_BUILDING;
  const coords = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    .map(([x, y]) => [longitude + x * 0.0000113, latitude + y * 0.000009]);
  return {
    type: 'Feature', properties: { kind, buildingId: id, baseM: kind === 'buildings' ? 0 : heightM + 0.12, topM: kind === 'buildings' ? heightM : heightM + 0.48 },
    geometry: { type: 'Polygon', coordinates: [coords] },
  };
}

const demoScene: FeatureCollection<Polygon> = {
  type: 'FeatureCollection', features: [
    rectangle('buildings', -42, -25, 42, 25),
    rectangle('shadows', -39, 5, -10, 22),
    rectangle('panels', -39, -21, -29, -4),
    rectangle('panels', -26, -21, -16, -4),
    rectangle('available', -3, -20, 37, 17),
  ],
};

export function syncDemoLayers(map: SceneMap, visibility: LayerVisibility) {
  if (!map.getLayer(layerKey('buildings'))) return;
  for (const { id } of LAYERS) map.setLayoutProperty(layerKey(id), 'visibility', visibility[id] ? 'visible' : 'none');
}

export function registerDemoScene(map: SceneMap, visibility: LayerVisibility) {
  if (map.getLayer(layerKey('buildings'))) return;
  map.addSource(SOURCE, { type: 'geojson', data: demoScene });
  for (const { id, color } of LAYERS) {
    map.addLayer({
      id: layerKey(id), type: 'fill-extrusion', source: SOURCE, minzoom: 16.4,
      filter: ['==', ['get', 'kind'], id], layout: { visibility: visibility[id] ? 'visible' : 'none' },
      paint: { 'fill-extrusion-color': color, 'fill-extrusion-base': ['get', 'baseM'], 'fill-extrusion-height': ['get', 'topM'] },
    });
  }
}

export function hitDemoBuilding(map: SceneMap, point: [number, number]) {
  if (map.getZoom() < 16.4 || !map.getLayer(layerKey('buildings'))) return false;
  return map.queryRenderedFeatures(point, { layers: LAYERS.map(({ id }) => layerKey(id)) })
    .some((feature) => feature.properties?.buildingId === DEMO_BUILDING.id);
}
