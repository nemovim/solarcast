import type { Feature, FeatureCollection, MultiPolygon, Polygon } from 'geojson';
import type { Viewport } from './map-panel';
import { extent, worldCoordinate, worldPixel } from './client-imagery';
import { booleanIntersects, polygon, featureCollection } from '@turf/turf';

type Tile = { file: string; count: number; processed_at: string };
export type PrecomputedManifest = { version: number; zoom: number; model: string; model_sha256: string; min_confidence?: number; source: string; imagery_observed_at: string | null; updated_at: string; tiles: Record<string, Tile> };
export type PanelFeature = Feature<Polygon | MultiPolygon, { confidence: number }>;
export type RoofFeature = Feature<Polygon | MultiPolygon>;
export type PrecomputedModel = { id: string; name: string };
export type ModelKind = 'panel' | 'roof';
// Manifests and validated tile features fetched during this page session, shared by the map
// previews and the analysis so a tile is downloaded once.
export type TileStore = { manifests: Map<string, Promise<PrecomputedManifest>>; tiles: Map<string, Promise<RoofFeature[]>> };
export const createTileStore = (): TileStore => ({ manifests: new Map(), tiles: new Map() });
const CONCURRENCY = 16;

function modelPath(kind: ModelKind, modelId: string): string {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(modelId)) throw new Error('사전 추론 모델 이름이 올바르지 않습니다.');
  return `/precomputed/${kind}/${encodeURIComponent(modelId)}`;
}

export async function readModels(kind: ModelKind): Promise<PrecomputedModel[]> {
  const response = await fetch(`/precomputed/${kind}/index.json`, { cache: 'no-store' });
  if (response.status === 404) return [];
  if (!response.ok) throw new Error('사전 추론 모델 목록을 불러오지 못했습니다.');
  const catalog = await response.json();
  if (catalog.version !== 1 || !Array.isArray(catalog.models) || catalog.models.some((model: PrecomputedModel) => !model || typeof model.id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(model.id) || typeof model.name !== 'string') || new Set(catalog.models.map((model: PrecomputedModel) => model.id)).size !== catalog.models.length) throw new Error('사전 추론 모델 목록 형식이 올바르지 않습니다.');
  return catalog.models;
}

export async function readManifest(kind: ModelKind, modelId: string): Promise<PrecomputedManifest> {
  const response = await fetch(`${modelPath(kind, modelId)}/manifest.json`, { cache: 'no-store' });
  if (response.status === 404) throw new Error('사전 추론 결과가 아직 등록되지 않았습니다.');
  if (!response.ok) throw new Error('사전 추론 목록을 불러오지 못했습니다.');
  const manifest: PrecomputedManifest = await response.json();
  if (manifest.version !== 1 || manifest.zoom !== 19 || typeof manifest.tiles !== 'object' || manifest.tiles === null || (kind === 'panel' && typeof manifest.min_confidence !== 'number')) throw new Error('사전 추론 목록 형식이 올바르지 않습니다.');
  return manifest;
}

function tileRange(viewport: Viewport) {
  const { west, south, east, north } = extent(viewport);
  const [left, top] = worldPixel(west, north), [right, bottom] = worldPixel(east, south);
  return { x0: Math.floor(left/256), x1: Math.floor(right/256), y0: Math.floor(top/256), y1: Math.floor(bottom/256) };
}

function tilePolygon(x: number, y: number) {
  const nw=worldCoordinate(x*256,y*256), ne=worldCoordinate((x+1)*256,y*256);
  const se=worldCoordinate((x+1)*256,(y+1)*256), sw=worldCoordinate(x*256,(y+1)*256);
  return polygon([[nw,ne,se,sw,nw]]);
}

export function viewportTiles(viewport: Viewport, limit = 1600): string[] {
  const { x0, x1, y0, y1 } = tileRange(viewport);
  if ((x1-x0+1)*(y1-y0+1)>limit) throw new Error('화면 범위가 너무 넓습니다. 사전 분석 구역으로 확대해 주세요.');
  const tiles: string[]=[];
  const region=polygon([[...viewport,viewport[0]]]);
  for (let x=x0;x<=x1;x++) for (let y=y0;y<=y1;y++) if (booleanIntersects(region,tilePolygon(x,y))) tiles.push(`${x}_${y}`);
  return tiles;
}

export function coverageGeometry(manifest: PrecomputedManifest, other?: PrecomputedManifest, selected?: Set<string>): FeatureCollection<Polygon> {
  const features: Feature<Polygon>[] = [];
  for (const key of Object.keys(manifest.tiles)) {
    if (other && !other.tiles[key]) continue;
    if (selected && !selected.has(key)) continue;
    if (!/^\d{1,6}_\d{1,6}$/.test(key)) continue;
    const [x,y] = key.split('_').map(Number);
    const nw=worldCoordinate(x*256,y*256), ne=worldCoordinate((x+1)*256,y*256);
    const se=worldCoordinate((x+1)*256,(y+1)*256), sw=worldCoordinate(x*256,(y+1)*256);
    features.push({type:'Feature',geometry:{type:'Polygon',coordinates:[[nw,ne,se,sw,nw]]},properties:{tileKey:key}});
  }
  return {type:'FeatureCollection',features};
}

export function cachedManifest(store: TileStore, kind: ModelKind, modelId: string): Promise<PrecomputedManifest> {
  const id = `${kind}/${modelId}`;
  let pending = store.manifests.get(id);
  if (!pending) {
    pending = readManifest(kind, modelId);
    store.manifests.set(id, pending);
    pending.catch(() => { if (store.manifests.get(id) === pending) store.manifests.delete(id); });
  }
  return pending;
}

// All validated features of one processed tile. Tiles recorded with no detections are not
// downloaded.
function tileFeatures(store: TileStore, kind: ModelKind, modelId: string, manifest: PrecomputedManifest, key: string): Promise<RoofFeature[]> {
  const entry = manifest.tiles[key];
  if (entry.file !== `tiles/${key}.geojson`) return Promise.reject(new Error('사전 추론 파일 경로가 올바르지 않습니다.'));
  if (entry.count === 0) return Promise.resolve([]);
  const id = `${kind}/${modelId}/${key}`;
  let pending = store.tiles.get(id);
  if (!pending) {
    pending = (async () => {
      const response=await fetch(`${modelPath(kind, modelId)}/${entry.file}`);
      if (!response.ok) throw new Error('사전 추론 파일이 누락되었습니다.');
      const data=await response.json();
      if (data.type!=='FeatureCollection' || !Array.isArray(data.features) || data.features.length!==entry.count) throw new Error('사전 추론 파일이 손상되었습니다.');
      for (const feature of data.features) {
        const score=feature?.properties?.confidence;
        if (feature?.type !== 'Feature' || !['Polygon','MultiPolygon'].includes(feature?.geometry?.type)) throw new Error('사전 추론 결과가 올바르지 않습니다.');
        if (kind === 'panel' && (typeof score !== 'number' || !Number.isFinite(score) || score<0 || score>1)) throw new Error('사전 추론 결과가 올바르지 않습니다.');
      }
      return data.features as RoofFeature[];
    })();
    store.tiles.set(id, pending);
    pending.catch(() => { if (store.tiles.get(id) === pending) store.tiles.delete(id); });
  }
  return pending;
}

async function loadTiles(kind: ModelKind, modelId: string, viewport: Viewport, threshold: number, progress: (s: string) => void, knownManifest?: PrecomputedManifest, selectedTiles?: string[], store = createTileStore()) {
  progress('사전 추론 결과를 불러오는 중입니다.');
  const manifest = knownManifest ?? await cachedManifest(store, kind, modelId);
  if (kind === 'panel' && threshold < manifest.min_confidence!) throw new Error(`사전 결과는 신뢰도 ${Math.round(manifest.min_confidence!*100)}% 이상부터 제공됩니다.`);
  const selected=selectedTiles ?? viewportTiles(viewport);
  const missing=selected.filter((key)=>!manifest.tiles[key]);
  if (missing.length) throw new Error(`화면의 ${missing.length}개 영역에 사전 추론 결과가 없습니다. 분석 가능 영역 안으로 이동하거나 확대해 주세요.`);
  const features: RoofFeature[]=[];
  let completed=0, next=0;
  async function worker() {
    while (next<selected.length) {
      for (const feature of await tileFeatures(store, kind, modelId, manifest, selected[next++]))
        if (kind === 'roof' || (feature as PanelFeature).properties.confidence >= threshold) features.push(feature);
      if (++completed % 20 === 0 || completed === selected.length) progress(`사전 추론 결과 불러오는 중 (${completed}/${selected.length})`);
    }
  }
  await Promise.all(Array.from({length:Math.min(CONCURRENCY,selected.length)},()=>worker()));
  return {features, manifest};
}

export async function loadPrecomputed(modelId: string, viewport: Viewport, threshold: number, progress: (s: string) => void) {
  const result = await loadTiles('panel', modelId, viewport, threshold, progress);
  return {panels: result.features as PanelFeature[], manifest: result.manifest};
}

export async function loadPanelPreview(modelId: string, viewport: Viewport) {
  const manifest = await readManifest('panel', modelId);
  const selected = viewportTiles(viewport).filter((key) => Boolean(manifest.tiles[key]));
  const result = await loadTiles('panel', modelId, viewport, manifest.min_confidence!, () => {}, manifest, selected);
  return {panels: result.features as PanelFeature[], minConfidence: manifest.min_confidence!};
}

// Each loader belongs to one selected model and returns every tile it has shown so far.
export function createPreviewLoader(kind: ModelKind, modelId: string, store = createTileStore()) {
  const shown = new Set<string>();
  return async (viewport: Viewport) => {
    const manifest = await cachedManifest(store, kind, modelId);
    const selected = viewportTiles(viewport).filter((key) => Boolean(manifest.tiles[key]));
    await loadTiles(kind, modelId, viewport, kind === 'panel' ? manifest.min_confidence! : 0, () => {}, manifest, selected, store);
    for (const key of selected) shown.add(key);
    const features = await Promise.all([...shown].map((key) => tileFeatures(store, kind, modelId, manifest, key)));
    return {features:features.flat(), minConfidence:manifest.min_confidence};
  };
}

export async function loadRoofs(modelId: string, viewport: Viewport, progress: (s: string) => void) {
  const result = await loadTiles('roof', modelId, viewport, 0, progress);
  return {roofs: result.features, manifest: result.manifest};
}

export async function loadAvailable(panelModel: string, roofModel: string, viewport: Viewport, threshold: number, progress: (s: string) => void, store = createTileStore()) {
  const [panelManifest, roofManifest] = await Promise.all([cachedManifest(store, 'panel', panelModel), cachedManifest(store, 'roof', roofModel)]);
  // Only processed tiles inside the viewport's tile range need a geometry test.
  const { x0, x1, y0, y1 } = tileRange(viewport);
  const region = polygon([[...viewport, viewport[0]]]);
  const selected = Object.keys(panelManifest.tiles).filter((key) => {
    if (!roofManifest.tiles[key] || !/^\d{1,6}_\d{1,6}$/.test(key)) return false;
    const [x, y] = key.split('_').map(Number);
    return x >= x0 && x <= x1 && y >= y0 && y <= y1 && booleanIntersects(region, tilePolygon(x, y));
  });
  if (!selected.length) throw new Error('화면에 두 모델의 사전 추론 결과가 겹치는 영역이 없습니다.');
  if (selected.length > 1600) throw new Error('분석 가능한 영역이 너무 넓습니다. 지도를 확대해 주세요.');
  // A viewport wider than four times the analysis limit cannot be fully covered.
  const partial = (x1-x0+1)*(y1-y0+1) > 6400 || viewportTiles(viewport, 6400).length > selected.length;
  const [panel, roof] = await Promise.all([
    loadTiles('panel', panelModel, viewport, threshold, progress, panelManifest, selected, store),
    loadTiles('roof', roofModel, viewport, 0, progress, roofManifest, selected, store),
  ]);
  return {panels: panel.features as PanelFeature[], roofs: roof.features, panelManifest, roofManifest,
          coverage: coverageGeometry(panelManifest, roofManifest, new Set(selected)), processed: selected.length, partial};
}
