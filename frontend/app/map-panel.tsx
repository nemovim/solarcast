'use client';

import { useEffect, useRef, useState } from 'react';
import { Box, Compass, Crosshair, Globe2, LocateFixed, Minus, Plus, X } from 'lucide-react';
import type { GeoJSONSource, Map as MapLibreMap, StyleSpecification } from 'maplibre-gl';
import { DEMO_BUILDING, hitDemoBuilding, registerDemoScene, syncDemoLayers, type LayerVisibility } from './demo-building';
import { BUILDING_LAYER, BUILDING_SOURCE, BUILDING_ZOOM, EMPTY_BUILDINGS, KOREA_BOUNDS, type BuildingCollection, type BuildingInfo } from './vworld-map';
import type { AnalysisResult } from './analysis';
import type { FeatureCollection, Polygon } from 'geojson';

export type MapPosition = { lng: number; lat: number; zoom: number };
export type Viewport = [number, number][];
export const INITIAL_POSITION = { lng: 126.7885, lat: 35.0218, zoom: 13.6 };
export const PLACES = [
  { name: '서울', lng: 126.9784, lat: 37.5667, zoom: 13.6 },
  { name: '부산', lng: 129.0756, lat: 35.1796, zoom: 13.6 },
  { name: '대전', lng: 127.3845, lat: 36.3504, zoom: 13.6 },
  { name: '나주', ...INITIAL_POSITION },
] as const;

const mapStyle = (vworld: boolean): StyleSpecification => ({
  version: 8,
  sources: { streets: {
    type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256,
    maxzoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }, ...(vworld ? { aerial: {
    type: 'raster' as const, tiles: [`${location.origin}/api/vworld/tiles/{z}/{x}/{y}`], tileSize: 256,
    minzoom: 5, maxzoom: 19, bounds: KOREA_BOUNDS, attribution: '© <a href="https://www.vworld.kr/">VWorld</a>',
  } } : {}) },
  layers: [
    { id: 'backdrop', type: 'background', paint: { 'background-color': '#28382d' } },
    { id: 'streets', type: 'raster', source: 'streets', layout: { visibility: vworld ? 'none' : 'visible' }, paint: { 'raster-saturation': -0.52, 'raster-contrast': 0.08, 'raster-brightness-max': 0.88 } },
    ...(vworld ? [{ id: 'aerial', type: 'raster' as const, source: 'aerial', paint: { 'raster-fade-duration': 0 } }] : []),
  ],
});

const demoNumber = new Intl.NumberFormat('ko-KR');
const heightNumber = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 1 });
const HIDDEN_DEMO: LayerVisibility = { buildings: false, shadows: false, panels: false, available: false };

function buildingAt(map: MapLibreMap, point: [number, number]): BuildingInfo | null {
  if (!map.getLayer(BUILDING_LAYER) || map.getZoom() < BUILDING_ZOOM) return null;
  const properties = map.queryRenderedFeatures(point, { layers: [BUILDING_LAYER] })[0]?.properties;
  if (!properties || typeof properties.heightM !== 'number' || typeof properties.name !== 'string') return null;
  return { name: properties.name, heightM: properties.heightM, heightSource: properties.heightSource };
}

export default function MapPanel({ target, onPosition, onViewport, visibility, vworldEnabled, analysis, coverage }: {
  target: MapPosition; onPosition: (position: MapPosition) => void; onViewport: (viewport: Viewport) => void; visibility: LayerVisibility; vworldEnabled: boolean; analysis: AnalysisResult | null; coverage: FeatureCollection<Polygon>;
}) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const pending = useRef(target);
  const realBuildingMode = useRef(visibility.buildings);
  const analysisRef = useRef(analysis);
  analysisRef.current = analysis;
  const coverageRef = useRef(coverage);
  coverageRef.current = coverage;
  const updateAnalysis = (activeMap: MapLibreMap, result: AnalysisResult | null) => {
    for (const [name, color] of [['available_roofs', '#18b776'], ['existing_panels', '#397ee8'], ['shadows', '#3b3d54']] as const) {
      const id = `analysis-${name}`;
      const data = result?.[name] ?? { type: 'FeatureCollection' as const, features: [] };
      const source = activeMap.getSource<GeoJSONSource>(id);
      if (source) source.setData(data);
      else {
        activeMap.addSource(id, { type: 'geojson', data });
        activeMap.addLayer({ id, source: id, type: 'fill', paint: { 'fill-color': color, 'fill-opacity': 0.65 } });
      }
    }
  };
  const demoButton = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const [state, setState] = useState<'loading' | 'ready' | 'unavailable'>('loading');
  const [streets, setStreets] = useState(!vworldEnabled);
  const [viewingDemo, setViewingDemo] = useState(false);
  const [tilted, setTilted] = useState(false);
  const [buildingCount, setBuildingCount] = useState(0);
  const [selection, setSelection] = useState<'demo' | BuildingInfo | null>(null);
  const [buildingStatus, setBuildingStatus] = useState<'zoom' | 'loading' | 'empty' | 'ready' | 'partial' | 'error'>('zoom');
  const popupOpen = selection !== null;
  const fromDemo = useRef(true);
  const mapElement = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let mounted = true;
    let map: MapLibreMap | undefined;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let delay: ReturnType<typeof setTimeout> | undefined;
    let buildingRequest: AbortController | undefined;
    let loadedBuildings: { bounds: [number, number, number, number]; count: number } | undefined;
    let usingAerial = vworldEnabled;
    let frameHasTiles = false;
    const unavailable = () => { if (mounted) setState('unavailable'); };
    const loadBuildings = (activeMap: MapLibreMap) => {
      clearTimeout(delay);
      buildingRequest?.abort();
      const source = activeMap.getSource<GeoJSONSource>(BUILDING_SOURCE);
      if (!source) return;
      const [[w, s], [e, n]] = activeMap.getBounds().toArray();
      const [west, south, east, north] = KOREA_BOUNDS;
      if (activeMap.getZoom() < BUILDING_ZOOM || e - w > 0.052 || n - s > 0.052) { setBuildingStatus('zoom'); return; }
      if (e <= west || w >= east || n <= south || s >= north) { setBuildingStatus('empty'); return; }
      const visible = [Math.max(west, w), Math.max(south, s), Math.min(east, e), Math.min(north, n)];
      if (loadedBuildings && visible[0] >= loadedBuildings.bounds[0] && visible[1] >= loadedBuildings.bounds[1] && visible[2] <= loadedBuildings.bounds[2] && visible[3] <= loadedBuildings.bounds[3]) {
        setBuildingCount(loadedBuildings.count);
        setBuildingStatus(loadedBuildings.count ? 'ready' : 'empty');
        return;
      }
      // Request a margin around the viewport so small movements need no new fetch.
      const xMargin = Math.max(0, Math.min((visible[2] - visible[0]) * 0.25, (0.054 - (visible[2] - visible[0])) / 2));
      const yMargin = Math.max(0, Math.min((visible[3] - visible[1]) * 0.25, (0.054 - (visible[3] - visible[1])) / 2));
      const bounds: [number, number, number, number] = [
        Math.max(west, Math.floor((visible[0] - xMargin) * 1e6) / 1e6),
        Math.max(south, Math.floor((visible[1] - yMargin) * 1e6) / 1e6),
        Math.min(east, Math.ceil((visible[2] + xMargin) * 1e6) / 1e6),
        Math.min(north, Math.ceil((visible[3] + yMargin) * 1e6) / 1e6),
      ];
      const bbox = bounds.map((coord) => coord.toFixed(6)).join(',');
      setBuildingStatus('loading');
      delay = setTimeout(async () => {
        const controller = new AbortController();
        buildingRequest = controller;
        try {
          const response = await fetch(`/api/vworld/buildings?bbox=${bbox}`, { signal: controller.signal });
          if (!response.ok) throw new Error('Building data unavailable');
          const data: BuildingCollection & { partial?: boolean } = await response.json();
          if (controller.signal.aborted || !mounted) return;
          if (data.type !== 'FeatureCollection' || !Array.isArray(data.features)) throw new Error('Invalid building data');
          source.setData(data);
          loadedBuildings = data.partial ? undefined : { bounds, count: data.features.length };
          setBuildingCount(data.features.length);
          setBuildingStatus(!data.features.length ? 'empty' : data.partial ? 'partial' : 'ready');
        } catch { if (!controller.signal.aborted && mounted) setBuildingStatus('error'); }
      }, 300);
    };

    void import('maplibre-gl').then((maplibregl) => {
      if (!mounted || !container.current) return;
      maplibregl.setWorkerUrl('/maplibre-gl-worker.mjs');
      const initial = pending.current;
      const activeMap = new maplibregl.Map({ container: container.current, style: mapStyle(vworldEnabled), center: [initial.lng, initial.lat], zoom: initial.zoom, minZoom: 5, maxZoom: 20, maxPitch: 65, renderWorldCopies: false, dragRotate: false, touchPitch: false });
      map = mapRef.current = activeMap;
      activeMap.touchZoomRotate.disableRotation();
      timeout = setTimeout(() => {
        if (!usingAerial) { unavailable(); return; }
        usingAerial = false;
        setStreets(true);
        activeMap.setLayoutProperty('aerial', 'visibility', 'none');
        activeMap.setLayoutProperty('streets', 'visibility', 'visible');
        timeout = setTimeout(unavailable, 9000);
      }, 7000);
      activeMap.on('sourcedata', (event) => {
        if (event.sourceId === (usingAerial ? 'aerial' : 'streets') && event.tile?.state === 'loaded') {
          frameHasTiles = true;
          clearTimeout(timeout);
          setState('ready');
        }
      });
      activeMap.on('error', () => { /* Show the fallback when the basemap fails to load. */ });
      let sceneRegistered = false;
      const updateViewport = () => {
        const canvas = activeMap.getCanvas();
        const corners: [number, number][] = [[0, 0], [canvas.clientWidth, 0], [canvas.clientWidth, canvas.clientHeight], [0, canvas.clientHeight]];
        onViewport(corners.map((point) => { const value = activeMap.unproject(point); return [value.lng, value.lat]; }));
      };
      const registerScene = () => {
        if (sceneRegistered || !activeMap.getLayer('streets')) return;
        sceneRegistered = true;
        if (vworldEnabled) {
          activeMap.addSource(BUILDING_SOURCE, { type: 'geojson', data: EMPTY_BUILDINGS, attribution: '© VWorld' });
          activeMap.addLayer({ id: BUILDING_LAYER, type: 'fill-extrusion', source: BUILDING_SOURCE, minzoom: BUILDING_ZOOM,
            layout: { visibility: realBuildingMode.current ? 'visible' : 'none' },
            paint: { 'fill-extrusion-color': '#9cafac', 'fill-extrusion-height': ['get', 'heightM'], 'fill-extrusion-opacity': 0.86 },
          });
        }
        activeMap.addSource('precomputed-coverage', { type: 'geojson', data: coverageRef.current });
        activeMap.addLayer({ id:'precomputed-coverage-outline', type:'line', source:'precomputed-coverage', paint:{ 'line-color':'#ffc45a', 'line-width':1.5, 'line-opacity':0.8 } });
        registerDemoScene(activeMap, HIDDEN_DEMO);
        updateAnalysis(activeMap, analysisRef.current);
        updateViewport();
        loadBuildings(activeMap);
      };
      activeMap.on('styledata', registerScene);
      activeMap.on('load', registerScene);
      activeMap.on('moveend', () => {
        const center = activeMap.getCenter();
        setTilted(activeMap.getPitch() > 0);
        onPosition({ lng: center.lng, lat: center.lat, zoom: activeMap.getZoom() });
        updateViewport();
        loadBuildings(activeMap);
      });
      activeMap.on('resize', () => { updateViewport(); loadBuildings(activeMap); });
      activeMap.on('click', ({ point }) => {
        if (hitDemoBuilding(activeMap, [point.x, point.y])) {
          setViewingDemo(true);
          fromDemo.current = true;
          setSelection('demo');
          return;
        }
        const building = buildingAt(activeMap, [point.x, point.y]);
        if (building) { fromDemo.current = false; setSelection(building); }
      });
      activeMap.on('mousemove', ({ point }) => {
        activeMap.getCanvas().style.cursor = frameHasTiles && (hitDemoBuilding(activeMap, [point.x, point.y]) || buildingAt(activeMap, [point.x, point.y])) ? 'pointer' : '';
      });
    }).catch(unavailable);
    return () => {
      mounted = false;
      clearTimeout(timeout);
      clearTimeout(delay);
      buildingRequest?.abort();
      map?.remove();
      mapRef.current = null;
    };
  }, [onPosition, onViewport, vworldEnabled]);

  useEffect(() => {
    mapRef.current?.getSource<GeoJSONSource>('precomputed-coverage')?.setData(coverage);
  }, [coverage]);

  useEffect(() => {
    const map = mapRef.current;
    if (analysis) { setViewingDemo(false); setSelection(null); }
    if (map?.isStyleLoaded()) updateAnalysis(map, analysis);
  }, [analysis]);

  useEffect(() => {
    pending.current = target;
    setViewingDemo(false);
    setSelection(null);
    mapRef.current?.jumpTo({ center: [target.lng, target.lat], zoom: target.zoom, pitch: 0, bearing: 0 });
  }, [target]);

  useEffect(() => {
    realBuildingMode.current = visibility.buildings && !viewingDemo;
    const map = mapRef.current;
    if (!map) return;
    syncDemoLayers(map, viewingDemo ? visibility : HIDDEN_DEMO);
    if (map.getLayer(BUILDING_LAYER)) map.setLayoutProperty(BUILDING_LAYER, 'visibility', realBuildingMode.current ? 'visible' : 'none');
    for (const [id, shown] of [['analysis-available_roofs', visibility.available], ['analysis-existing_panels', visibility.panels], ['analysis-shadows', visibility.shadows]] as const) {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', shown && !viewingDemo ? 'visible' : 'none');
    }
  }, [visibility, viewingDemo, analysis]);

  useEffect(() => { if (popupOpen) closeButton.current?.focus({ preventScroll: true }); }, [popupOpen]);

  const moveZoom = (delta: number) => {
    const map = mapRef.current;
    if (map) map.jumpTo({ zoom: map.getZoom() + delta });
  };

  const showDemo = () => {
    const map = mapRef.current;
    if (!map) return;
    map.jumpTo({ center: [DEMO_BUILDING.longitude, DEMO_BUILDING.latitude], zoom: 17.5, pitch: 58, bearing: -24 });
    setViewingDemo(true);
    fromDemo.current = true;
    setSelection('demo');
  };

  const closePopup = () => {
    setSelection(null);
    if (fromDemo.current) {
      setViewingDemo(false);
      demoButton.current?.focus({ preventScroll: true });
    }
    else if (mapElement.current) mapElement.current.focus({ preventScroll: true });
    else mapRef.current?.getCanvas().focus({ preventScroll: true });
  };
  const actual = selection && selection !== 'demo' ? selection : null;
  const buildingStatusText = {
    zoom: '더 확대하면 실제 건물을 볼 수 있어요.', loading: '브이월드 건물을 불러오는 중…',
    empty: '이 범위에는 표시할 건물이 없어요.', ready: `건물 ${demoNumber.format(buildingCount)}개 불러옴 · ${tilted ? '건물을 눌러 보세요' : '3D 보기로 높이를 확인하세요'}`,
    partial: `건물 ${demoNumber.format(buildingCount)}개 불러옴 · 더 확대해 주세요`, error: '건물을 불러오지 못했어요. 지도를 이동해 다시 조회해 주세요.',
  }[buildingStatus];

  return (
    <div className="map-frame relative isolate min-w-0 overflow-hidden" aria-label="분석 위치 지도">
      <div className="map-canvas absolute inset-0" ref={container} />
      <div className="map-tools absolute top-4 right-4 flex flex-col gap-2">
        <div className="zoom-buttons">
          <button type="button" onClick={() => moveZoom(1)} disabled={state !== 'ready'} aria-label="지도 확대"><Plus size={18} aria-hidden="true" /></button>
          <button type="button" onClick={() => moveZoom(-1)} disabled={state !== 'ready'} aria-label="지도 축소"><Minus size={18} aria-hidden="true" /></button>
        </div>
        <button type="button" disabled={state !== 'ready'} onClick={() => {
          setViewingDemo(false); setSelection(null);
          mapRef.current?.jumpTo({ center: [target.lng, target.lat], zoom: target.zoom, pitch: 0, bearing: 0 });
        }} aria-label="선택한 도시로 돌아가기"><LocateFixed size={17} aria-hidden="true" /></button>
        {vworldEnabled && <button type="button" className="map-view-mode" disabled={state !== 'ready'} aria-label={tilted && !viewingDemo ? '2D 보기' : '3D 보기'} onClick={() => {
          const map = mapRef.current;
          if (!map) return;
          setViewingDemo(false);
          setSelection(null);
          if (tilted && !viewingDemo) map.jumpTo({ pitch: 0 });
          else map.jumpTo({ zoom: Math.max(map.getZoom(), 17.5), pitch: 52 });
        }}><Box size={15} aria-hidden="true" />{tilted && !viewingDemo ? '2D' : '3D'}</button>}
      </div>
      <div className="map-tag absolute top-4 left-4 flex items-center gap-2"><Globe2 size={14} aria-hidden="true" />{streets ? (vworldEnabled ? '항공사진 연결 실패 · 기본 지도' : '기본 지도 · OSM') : '항공사진 · 브이월드'}</div>
      <button ref={demoButton} type="button" className="map-demo-button" onClick={showDemo} disabled={state !== 'ready'} aria-expanded={popupOpen} aria-controls={popupOpen ? 'building-result' : undefined}><Box size={17} aria-hidden="true" /> 3D 건물 시연</button>
      {vworldEnabled && !viewingDemo && visibility.buildings && (buildingStatus === 'ready' || buildingStatus === 'partial') && <button ref={mapElement} type="button" className="map-center-info" aria-label="지도 중심 건물 정보" onClick={() => {
        const map = mapRef.current;
        if (!map) return;
        const canvas = map.getCanvas();
        const building = buildingAt(map, [canvas.clientWidth / 2, canvas.clientHeight / 2]);
        fromDemo.current = false;
        setSelection(building ?? { name: '지도 중심에 건물이 없어요', heightM: 0, heightSource: 'default' });
      }}><LocateFixed size={14} aria-hidden="true" /> 중심 건물 정보</button>}
      {state !== 'ready' && (
        <div className="map-message" role="status">
          <Compass size={28} aria-hidden="true" />
          <strong>{state === 'loading' ? '지도를 불러오는 중…' : '지도를 불러오지 못했어요'}</strong>
          <span>{state === 'loading' ? '지도 연결을 확인하고 있어요.' : '연결 또는 브라우저의 WebGL 지원을 확인해 주세요.'}</span>
        </div>
      )}
      {!popupOpen && <div className="map-caption absolute left-4 bottom-5" role="status"><Crosshair size={15} aria-hidden="true" /> {viewingDemo ? '시연 건물을 누르면 결과가 열려요' : !vworldEnabled ? '브이월드 미연결 · 기본 지도' : !visibility.buildings ? '3D 건물 레이어가 꺼져 있어요.' : buildingStatusText}</div>}
      {popupOpen && (
        <section id="building-result" className="building-popup" role="dialog" aria-labelledby="building-result-title" aria-describedby="building-result-note" onKeyDown={(event) => { if (event.key === 'Escape') closePopup(); }}>
          <div className="building-popup-heading flex items-start justify-between gap-3">
            <div>
              <p className="building-sample"><Box size={14} aria-hidden="true" /> {actual ? '브이월드 건물 정보' : '분석 결과 예시'}</p>
              <h3 id="building-result-title">{actual?.name ?? DEMO_BUILDING.name}</h3>
            </div>
            <button ref={closeButton} type="button" aria-label="건물 분석 결과 닫기" onClick={closePopup}><X size={18} aria-hidden="true" /></button>
          </div>
          <p id="building-result-note">{actual ? actual.heightM ? (analysis ? '분석 수치는 오른쪽 결과 목록에서 확인하세요.' : '이 건물의 분석 결과가 아직 없어요.') : '지도를 움직이거나 화면의 건물을 직접 눌러 주세요.' : '실제 건물이나 분석 결과가 아니에요.'}</p>
          {actual ? actual.heightM > 0 && <dl className="building-metrics"><div><dt>{actual.heightSource === 'record' ? '제공 높이' : actual.heightSource === 'floors' ? '층수 기반 추정 높이' : '높이 정보 없음 · 표시용 높이'}</dt><dd>{heightNumber.format(actual.heightM)} <small>m</small></dd></div></dl> : <dl className="building-metrics">
            <div><dt>설치 가능 면적</dt><dd>{demoNumber.format(DEMO_BUILDING.availableAreaM2)} <small>㎡</small></dd></div>
            <div><dt>설치 용량</dt><dd>{demoNumber.format(DEMO_BUILDING.capacityKw)} <small>kW</small></dd></div>
            <div><dt>예상 연간 발전량</dt><dd>{demoNumber.format(DEMO_BUILDING.annualEnergyKwh)} <small>kWh/년</small></dd></div>
          </dl>}
        </section>
      )}
    </div>
  );
}
