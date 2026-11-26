import type { Feature, FeatureCollection, MultiPolygon, Polygon } from 'geojson';
import type { Viewport } from './map-panel';
import { extent, worldCoordinate, worldPixel } from './client-imagery';
import { booleanIntersects, polygon, union, difference, featureCollection, area, intersect } from '@turf/turf';

type Tile = { file: string; count: number; processed_at: string };
export type PrecomputedManifest = { version: number; zoom: number; model: string; model_sha256: string; min_confidence?: number; source: string; imagery_observed_at: string | null; updated_at: string; tiles: Record<string, Tile> };
export type PanelFeature = Feature<Polygon | MultiPolygon, { confidence: number }>;
export type RoofFeature = Feature<Polygon | MultiPolygon>;
export type PrecomputedModel = { id: string; name: string };
export type ModelKind = 'panel' | 'roof';

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

export function viewportTiles(viewport: Viewport): string[] {
  const { west, south, east, north } = extent(viewport);
  const [left, top] = worldPixel(west, north), [right, bottom] = worldPixel(east, south);
  const x0=Math.floor(left/256), x1=Math.floor(right/256), y0=Math.floor(top/256), y1=Math.floor(bottom/256);
  if ((x1-x0+1)*(y1-y0+1)>1600) throw new Error('화면 범위가 너무 넓습니다. 사전 분석 구역으로 확대해 주세요.');
  const tiles: string[]=[];
  const region=polygon([[...viewport,viewport[0]]]);
  for (let x=x0;x<=x1;x++) for (let y=y0;y<=y1;y++) {
    const nw=worldCoordinate(x*256,y*256), ne=worldCoordinate((x+1)*256,y*256);
    const se=worldCoordinate((x+1)*256,(y+1)*256), sw=worldCoordinate(x*256,(y+1)*256);
    if (booleanIntersects(region,polygon([[nw,ne,se,sw,nw]]))) tiles.push(`${x}_${y}`);
  }
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

async function loadTiles(kind: ModelKind, modelId: string, viewport: Viewport, threshold: number, progress: (s: string) => void, knownManifest?: PrecomputedManifest, selectedTiles?: string[]) {
  progress('사전 추론 결과를 불러오는 중입니다.');
  const manifest = knownManifest ?? await readManifest(kind, modelId);
  if (kind === 'panel' && threshold < manifest.min_confidence!) throw new Error(`사전 결과는 신뢰도 ${Math.round(manifest.min_confidence!*100)}% 이상부터 제공됩니다.`);
  const selected=selectedTiles ?? viewportTiles(viewport);
  const missing=selected.filter((key)=>!manifest.tiles[key]);
  if (missing.length) throw new Error(`화면의 ${missing.length}개 영역에 사전 추론 결과가 없습니다. 분석 가능 영역 안으로 이동하거나 확대해 주세요.`);
  const features: RoofFeature[]=[];
  let completed=0, next=0;
  async function worker() {
    while (next<selected.length) {
      const key=selected[next++], entry=manifest.tiles[key];
      if (entry.file !== `tiles/${key}.geojson`) throw new Error('사전 추론 파일 경로가 올바르지 않습니다.');
      const response=await fetch(`${modelPath(kind, modelId)}/${entry.file}`);
      if (!response.ok) throw new Error('사전 추론 파일이 누락되었습니다.');
      const data=await response.json();
      if (data.type!=='FeatureCollection' || !Array.isArray(data.features) || data.features.length!==entry.count) throw new Error('사전 추론 파일이 손상되었습니다.');
      for (const feature of data.features) {
        const score=feature?.properties?.confidence;
        if (feature?.type !== 'Feature' || !['Polygon','MultiPolygon'].includes(feature?.geometry?.type)) throw new Error('사전 추론 결과가 올바르지 않습니다.');
        if (kind === 'panel' && (typeof score !== 'number' || !Number.isFinite(score) || score<0 || score>1)) throw new Error('사전 추론 결과가 올바르지 않습니다.');
        if (kind === 'roof' || score >= threshold) features.push(feature as RoofFeature);
      }
      progress(`사전 추론 결과 불러오는 중 (${++completed}/${selected.length})`);
    }
  }
  await Promise.all(Array.from({length:Math.min(4,selected.length)},()=>worker()));
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

// Each loader belongs to one selected model and keeps fetched tiles across map moves.
export function createPreviewLoader(kind: ModelKind, modelId: string) {
  let manifestPromise: Promise<PrecomputedManifest> | undefined;
  const tiles = new Map<string, Promise<RoofFeature[]>>();
  return async (viewport: Viewport) => {
    if (!manifestPromise) manifestPromise = readManifest(kind, modelId).catch((error) => { manifestPromise=undefined; throw error; });
    const manifest = await manifestPromise;
    const selected = viewportTiles(viewport).filter((key) => Boolean(manifest.tiles[key]));
    let next=0;
    async function worker() {
      while (next < selected.length) {
        const key=selected[next++];
        let pending=tiles.get(key);
        if (!pending) {
          pending=loadTiles(kind,modelId,viewport,kind === 'panel' ? manifest.min_confidence! : 0,()=>{},manifest,[key]).then((result)=>result.features);
          tiles.set(key,pending);
          pending.catch(()=>{ if (tiles.get(key) === pending) tiles.delete(key); });
        }
        await pending;
      }
    }
    await Promise.all(Array.from({length:Math.min(4,selected.length)},()=>worker()));
    return {features:(await Promise.all([...tiles.values()])).flat(), minConfidence:manifest.min_confidence};
  };
}

export async function loadRoofs(modelId: string, viewport: Viewport, progress: (s: string) => void) {
  const result = await loadTiles('roof', modelId, viewport, 0, progress);
  return {roofs: result.features, manifest: result.manifest};
}

export async function loadAvailable(panelModel: string, roofModel: string, viewport: Viewport, threshold: number, progress: (s: string) => void) {
  const [panelManifest, roofManifest] = await Promise.all([readManifest('panel', panelModel), readManifest('roof', roofModel)]);
  const region = polygon([[...viewport, viewport[0]]]);
  const common = coverageGeometry(panelManifest, roofManifest);
  const visible = common.features.filter((tile) => booleanIntersects(region, tile) && Boolean(intersect(featureCollection([region, tile]))));
  const selected = visible.map((tile) => String(tile.properties?.tileKey));
  if (!selected.length) throw new Error('화면에 두 모델의 사전 추론 결과가 겹치는 영역이 없습니다.');
  if (selected.length > 1600) throw new Error('분석 가능한 영역이 너무 넓습니다. 지도를 확대해 주세요.');
  const merged = visible.length === 1 ? visible[0] : union(featureCollection(visible));
  const remainder = merged ? difference(featureCollection([region, merged])) : region;
  const partial = Boolean(remainder && area(remainder) > 0.01);
  const [panel, roof] = await Promise.all([
    loadTiles('panel', panelModel, viewport, threshold, progress, panelManifest, selected),
    loadTiles('roof', roofModel, viewport, 0, progress, roofManifest, selected),
  ]);
  return {panels: panel.features as PanelFeature[], roofs: roof.features, panelManifest, roofManifest,
          coverage: featureCollection(visible), processed: selected.length, partial};
}
