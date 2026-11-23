// 발표용 사전 계산: 하루(기본 11-11)의 5분 간격 모든 시각에 대해 "현재 화면 건물 분석"이
// 하는 계산을 빛가람동 전체에 미리 돌려 public/precomputed/<MM-DD>.json 에 저장한다.
// 브라우저는 날짜·시각이 맞으면 이 결과를 그대로 읽어 그리고, 아니면 평소처럼 계산한다.
//
//   node scripts/precompute_day.mjs            # 11-11
//   node scripts/precompute_day.mjs 12-21      # 다른 날짜
//   node scripts/precompute_day.mjs 11-11 6    # 작업 스레드 수 (기본 4)
//
// 계산 코드는 src/shadow-core.js 를 그대로 쓰므로 브라우저 계산과 규칙이 같다.
// 다른 점은 판정 범위뿐이다 — 브라우저는 화면 + 여유분 건물만 보지만, 여기서는
// 지역 전체 건물을 그림자 원인으로 넣는다(화면 가장자리 판정이 오히려 더 정확하다).
// 시각(과 연간 표본)을 작업 스레드에 나눠 돌린다. 스레드마다 데이터를 따로 읽는다.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import SunCalc from 'suncalc';
import * as turf from '@turf/turf';
import {
  CENTER, MIN_SHADOW_AREA_RATIO, FLOOR_HEIGHT, FALLBACK_HEIGHT, PANEL_CELL_DEG,
  polygonFromFeature, measure, shadeOn, buildBuildingGrid, findHost, unleanGeometry,
  annualSamples,
} from './shadow-core.js';

const SELF = fileURLToPath(import.meta.url);
const DIR = path.dirname(SELF);
const KDN = path.resolve(DIR, '..');
const STEP_MIN = 5; // index.html timeSlider step 과 같아야 한다
const KST_OFFSET_H = 9; // 슬라이더 시각은 한국 표준시

// 표준 산출물 outputs/ 경로 매핑
const PANEL_FILES = {
  v3: 'outputs/3_panels/panels_v3.geojson',
  v2full: 'outputs/3_panels/panels_masking_v2.geojson',
  aihub: 'outputs/3_panels/panels_final.geojson',
  overfit: 'outputs/3_panels/panels_overfit.geojson',
};
const AVAIL_FILE = 'outputs/2_roof/panel_layout.geojson';
const read = (rel) => JSON.parse(fs.readFileSync(path.join(KDN, rel), 'utf8'));
const r3 = (v) => Math.round(v * 1000) / 1000;
const heightOf = (p) => {
  const floors = parseInt(p.gro_flo_co, 10) || 0;
  return floors > 0 ? floors * FLOOR_HEIGHT : FALLBACK_HEIGHT;
};

// ---- 공통 준비: main.js loadBuildings / collectBuildings / analyzePanels / runAnnual 과 같은 규칙 ----
function prepare() {
  const bldgFC = read('data/bitgaram/vector/buildings.geojson');
  const buildings = [];
  bldgFC.features.forEach((f, i) => {
    const p = f.properties;
    const height = heightOf(p);
    const bid = f.id || p.bd_mgt_sn || 'B' + i;
    polygonFromFeature(f).forEach((part, k) => {
      const b = { id: 'B' + buildings.length, idx: buildings.length, key: i + '.' + k, bid, groupId: bid, footprint: part, height };
      measure(b);
      buildings.push(b);
    });
  });
  const grid = buildBuildingGrid(buildings);

  const panelSets = {};
  for (const [key, rel] of Object.entries(PANEL_FILES)) {
    const fc = read(rel);
    const items = [];
    for (const f of fc.features) {
      const parts = polygonFromFeature(f);
      if (parts.length === 0) continue;
      const surface = parts.length === 1
        ? parts[0]
        : parts.reduce((a, b) => (turf.area(b) > turf.area(a) ? b : a));
      const panel = { id: f.properties.id, footprint: surface };
      measure(panel);
      const host = findHost(grid, panel.center);
      panel.height = host ? host.height : 0;
      if (host) {
        panel.footprint = turf.feature(unleanGeometry(surface.geometry, host.height));
        measure(panel);
      }
      items.push({ panel, hostBid: host ? host.bid : '' });
    }
    panelSets[key] = { count: fc.features.length, items };
  }

  const avail = read(AVAIL_FILE);
  const bldgByRid = new Map();
  bldgFC.features.forEach((f) => { if (f.id) bldgByRid.set(String(f.id).split('.').pop(), f); });
  const targets = [];
  avail.features.forEach((f, i) => {
    const host = bldgByRid.get(String(f.properties.building_id));
    if (!host) return;
    polygonFromFeature(f).forEach((part, k) => {
      const t = { key: i + '.' + k, footprint: part, height: heightOf(host.properties), sun: 0 };
      measure(t);
      targets.push(t);
    });
  });

  return { bldgFC, buildings, grid, panelSets, avail, targets };
}

// 그림자가 닿을 수 있는 건물만 골라 shadeOn 에 넘긴다. shadeOn 자체도 같은 거리 조건으로
// 건너뛰므로 결과는 전체를 넘길 때와 같다 — 원래 순서(idx)도 지켜 겹침 상한 동작까지 같게 한다.
function makeCasterPicker(buildings, grid) {
  const maxH = Math.max(...buildings.map((b) => b.height));
  const maxR = Math.max(...buildings.map((b) => b.radiusKm));
  const cellKm = PANEL_CELL_DEG * 88; // 경도 한 칸 ≈ 45m (위도 35°), 위도 한 칸 ≈ 55m — 작은 쪽
  return (target, ratioKm) => {
    const reachKm = Math.max(0, maxH - target.height) * ratioKm + maxR + target.radiusKm;
    const span = Math.ceil(reachKm / cellKm) + 1;
    const gx = Math.floor(target.center[0] / PANEL_CELL_DEG);
    const gy = Math.floor(target.center[1] / PANEL_CELL_DEG);
    const seen = new Set();
    for (let dx = -span; dx <= span; dx++) {
      for (let dy = -span; dy <= span; dy++) {
        for (const b of grid.get((gx + dx) + ':' + (gy + dy)) || []) seen.add(b);
      }
    }
    return [...seen].sort((a, b) => a.idx - b.idx);
  };
}

function runWorker({ md, stepMins, sampleIdx }) {
  const { buildings, grid, panelSets, targets } = prepare();
  const pick = makeCasterPicker(buildings, grid);
  const year = new Date().getFullYear();
  const [mo, d] = md.split('-').map(Number);

  for (const mins of stepMins) {
    const dt = new Date(Date.UTC(year, mo - 1, d, Math.floor(mins / 60) - KST_OFFSET_H, mins % 60));
    const pos = SunCalc.getPosition(dt, CENTER[1], CENTER[0]);
    const bearing = (((pos.azimuth * 180 / Math.PI) % 360) + 360) % 360;
    const ratioKm = (1 / Math.tan(pos.altitude)) / 1000;
    for (const b of buildings) b.shadowCache = new Map();

    const shaded = [];
    for (const b of buildings) {
      if (shadeOn(b, pick(b, ratioKm), bearing, ratioKm, MIN_SHADOW_AREA_RATIO).frac >= MIN_SHADOW_AREA_RATIO) {
        shaded.push(b.key);
      }
    }
    const panels = {};
    for (const [key, set] of Object.entries(panelSets)) {
      const fr = {};
      for (const { panel } of set.items) {
        const frac = shadeOn(panel, pick(panel, ratioKm), bearing, ratioKm, null).frac;
        if (frac > 0) fr[panel.id] = r3(frac);
      }
      panels[key] = fr;
    }
    parentPort.postMessage({ type: 'step', mins, data: { shaded, panels } });
  }

  const samples = annualSamples();
  for (const n of sampleIdx) {
    const { bearing, ratioKm, w } = samples[n];
    for (const b of buildings) b.shadowCache = new Map();
    const sun = {};
    for (const t of targets) sun[t.key] = w * (1 - shadeOn(t, pick(t, ratioKm), bearing, ratioKm, null).frac);
    parentPort.postMessage({ type: 'sample', n, w, sun });
  }
  parentPort.postMessage({ type: 'done' });
}

async function main() {
  const md = process.argv[2] || '11-11';
  const nWorkers = +(process.argv[3] || 4);
  const t0 = Date.now();
  const log = (msg) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s] ${msg}`);

  const { bldgFC, buildings, panelSets, avail, targets } = prepare();
  log(`건물 ${bldgFC.features.length}동 (조각 ${buildings.length}) · 설치 배열 ${targets.length}`
    + ` · 패널 ${Object.entries(panelSets).map(([k, s]) => `${k} ${s.count}`).join(', ')}`);

  const year = new Date().getFullYear();
  const [mo, d] = md.split('-').map(Number);
  const allSteps = [];
  for (let mins = 0; mins < 1440; mins += STEP_MIN) {
    const dt = new Date(Date.UTC(year, mo - 1, d, Math.floor(mins / 60) - KST_OFFSET_H, mins % 60));
    if (SunCalc.getPosition(dt, CENTER[1], CENTER[0]).altitude > 0) allSteps.push(mins);
  }
  const nSamples = annualSamples().length;
  log(`해 뜬 시각 ${allSteps.length}개 · 연간 표본 ${nSamples}개 · 스레드 ${nWorkers}`);

  // 해가 낮은 아침·저녁이 무거우므로 번갈아 나눠 스레드 부하를 고르게 한다
  const jobs = Array.from({ length: nWorkers }, (_, w) => ({
    md,
    stepMins: allSteps.filter((_, i) => i % nWorkers === w),
    sampleIdx: [...Array(nSamples).keys()].filter((i) => i % nWorkers === w),
  }));

  const steps = {};
  const sunSum = {};
  let wSum = 0;
  let doneSteps = 0;
  let doneSamples = 0;
  await Promise.all(jobs.map((job) => new Promise((resolve, reject) => {
    const worker = new Worker(SELF, { workerData: job });
    worker.on('message', (m) => {
      if (m.type === 'step') {
        steps[m.mins] = m.data;
        doneSteps++;
        const hh = String(Math.floor(m.mins / 60)).padStart(2, '0');
        const mm = String(m.mins % 60).padStart(2, '0');
        log(`시각 ${doneSteps}/${allSteps.length} (${hh}:${mm} 그림자 건물 ${m.data.shaded.length})`);
      } else if (m.type === 'sample') {
        wSum += m.w;
        for (const [k, v] of Object.entries(m.sun)) sunSum[k] = (sunSum[k] || 0) + v;
        doneSamples++;
        log(`연간 표본 ${doneSamples}/${nSamples}`);
      } else if (m.type === 'done') {
        resolve();
      }
    });
    worker.on('error', reject);
    worker.on('exit', (code) => { if (code !== 0) reject(new Error('worker exit ' + code)); });
  })));

  const annual = {};
  for (const t of targets) annual[t.key] = r3(wSum > 0 ? sunSum[t.key] / wSum : 1);
  const sortedSteps = Object.fromEntries(Object.keys(steps).map(Number).sort((a, b) => a - b)
    .map((k) => [k, steps[k]]));

  const out = {
    date: md,
    year,
    stepMin: STEP_MIN,
    createdAt: new Date().toISOString(),
    // 브라우저가 같은 데이터로 만든 결과인지 확인하는 데 쓴다 — 다르면 사전 계산을 쓰지 않는다
    counts: {
      buildings: bldgFC.features.length,
      avail: avail.features.length,
      panels: Object.fromEntries(Object.entries(panelSets).map(([k, s]) => [k, s.count])),
    },
    panelHosts: Object.fromEntries(Object.entries(panelSets).map(([k, s]) => [k,
      Object.fromEntries(s.items.map(({ panel, hostBid }) => [panel.id, [hostBid, r3(panel.height)]]))])),
    steps: sortedSteps,
    annual,
  };
  const saveDirs = [
    path.join(DIR, 'output'),
    path.join(KDN, 'outputs', '4_simulation'),
    path.join(KDN, '5_web_service', 'public', 'precomputed'),
  ];
  for (const d of saveDirs) {
    if (!fs.existsSync(d)) fs.mkdirSync(d, { recursive: true });
    const targetFile = path.join(d, (d.includes('4_simulation') ? `shadow_${md}.json` : `${md}.json`));
    fs.writeFileSync(targetFile, JSON.stringify(out));
    log(`저장: ${path.relative(KDN, targetFile)} (${(fs.statSync(targetFile).size / 1e6).toFixed(1)} MB)`);
  }
}

if (isMainThread) {
  main().catch((e) => { console.error(e); process.exit(1); });
} else {
  runWorker(workerData);
}
