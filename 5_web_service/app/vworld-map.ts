import type { FeatureCollection, MultiPolygon, Polygon } from 'geojson';

export const KOREA_BOUNDS: [number, number, number, number] = [123, 32, 133, 40];
export const BUILDING_ZOOM = 16.5;
export const BUILDING_SOURCE = 'vworld-footprints';
export const BUILDING_LAYER = 'vworld-3d-buildings';
export type BuildingInfo = { name: string; heightM: number; heightSource: 'record' | 'floors' | 'default' };
export type BuildingCollection = FeatureCollection<Polygon | MultiPolygon, BuildingInfo>;
export const EMPTY_BUILDINGS: BuildingCollection = { type: 'FeatureCollection', features: [] };
