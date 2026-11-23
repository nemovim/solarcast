// 그림자·음영 계산의 순수 함수와 상수. main.js(브라우저)와
// scripts/precompute_day.mjs(Node, 발표용 사전 계산)가 같은 코드를 쓴다.
import * as turf from '@turf/turf';
import SunCalc from 'suncalc';

export const CENTER = [126.7891, 35.0176]; // 나주 빛가람동 (혁신도시) 대략 중심

// "의미 있게" 그림자졌다고 볼 최소 면적 비율. 이보다 작은 스침은 무시.
// 건물 옥상과 태양광 패널 판정에 같은 기준을 쓴다.
export const MIN_SHADOW_AREA_RATIO = 0.1; // 10%
// 판정면보다 이만큼은 높아야 그 면에 그림자를 드리울 수 있다고 본다.
// 타일 경계로 쪼개진 같은 아파트 조각들(높이가 같다)도 이 기준에서 걸러진다.
export const MIN_HEIGHT_DIFF = 1; // m

export const PANEL_CELL_DEG = 0.0005; // 건물 격자 색인 한 칸 ≈ 55m (z19 타일 한 장 크기)
export const HOST_SNAP_KM = 0.008; // 바닥면 안에 안 들어오면 8m 까지는 가장 가까운 건물에 붙인다
export const MAX_OVERLAP_PIECES = 8; // 합집합에 넣을 겹침 폴리곤 상한

// VWorld 건물은 높이 대신 지상 층수(gro_flo_co)만 있다. 층고를 곱해 높이로 쓴다.
export const FLOOR_HEIGHT = 3.3; // m
export const FALLBACK_HEIGHT = 15; // m — 층수가 0/빈 값인 건물(47동)에 쓰는 높이

// ---- 시차(기복 변위) 보정 ----
// 영상은 정사영상이 아니라 높이 h 인 지붕이 바닥 외곽선에서 h × 비율 만큼 한 방향으로
// 밀려 찍혀 있다. 패널·설치가능 영역은 그 영상에서 나왔으므로 2D 에서는 맞지만,
// 그대로 옥상 높이로 올리면 건물 옆 허공에 뜬다. 호스트 높이만큼 되돌려서 올린다.
// 방향·비율은 지역 전체에 하나다. 시연에서는 꺼 두고(비율 0) 화면에서도 뺐다 —
// 건물별 변위를 추정하게 되면 이 두 값 대신 그 값을 쓴다.
export const LEAN_BEARING = 0; // 지붕이 밀린 방향, 북쪽 0° 시계방향
export const LEAN_RATIO = 0; // 높이 1m 당 밀림(m). 0 이면 보정하지 않는다
export const M_PER_DEG_LAT = 110574;
export const M_PER_DEG_LON = 111320 * Math.cos(CENTER[1] * Math.PI / 180);

export function unleanDelta(h) {
  const d = h * LEAN_RATIO;
  const rad = LEAN_BEARING * Math.PI / 180;
  return [-d * Math.sin(rad) / M_PER_DEG_LON, -d * Math.cos(rad) / M_PER_DEG_LAT];
}

export function shiftCoords(c, dLon, dLat) {
  return typeof c[0] === 'number'
    ? [c[0] + dLon, c[1] + dLat]
    : c.map((x) => shiftCoords(x, dLon, dLat));
}

export function unleanGeometry(geom, h) {
  const [dLon, dLat] = unleanDelta(h);
  if (dLon === 0 && dLat === 0) return geom;
  return { type: geom.type, coordinates: shiftCoords(geom.coordinates, dLon, dLat) };
}

export function unleanPoint(pt, h) {
  const [dLon, dLat] = unleanDelta(h);
  return [pt[0] + dLon, pt[1] + dLat];
}

export function polygonFromFeature(f) {
  // Polygon / MultiPolygon 건물 바닥면을 Polygon 배열로 정규화
  if (f.geometry.type === 'Polygon') {
    return [turf.polygon(f.geometry.coordinates)];
  } else if (f.geometry.type === 'MultiPolygon') {
    return f.geometry.coordinates.map((c) => turf.polygon(c));
  }
  return [];
}

export function computeShadowPolygon(footprint, bearingDeg, distanceKm) {
  try {
    const translated = turf.transformTranslate(footprint, distanceKm, bearingDeg, { units: 'kilometers' });
    const pts = [];
    turf.coordEach(footprint, (c) => pts.push(turf.point(c)));
    turf.coordEach(translated, (c) => pts.push(turf.point(c)));
    return turf.convex(turf.featureCollection(pts));
  } catch (e) {
    return null;
  }
}

export function measure(b) {
  b.center = turf.centroid(b.footprint).geometry.coordinates;
  b.area = turf.area(b.footprint) || 1;
  b.bbox = turf.bbox(b.footprint);
  let r = 0;
  turf.coordEach(b.footprint, (c) => {
    const d = turf.distance(b.center, c, { units: 'kilometers' });
    if (d > r) r = d;
  });
  b.radiusKm = r;
  b.shadowCache = new Map();
}

export function shadowAt(b, dh, bearingDeg, ratioKm) {
  const key = Math.round(dh * 2); // 0.5m 단위로 묶어 폴리곤 연산을 재사용
  if (b.shadowCache.has(key)) return b.shadowCache.get(key);
  const poly = key > 0
    ? computeShadowPolygon(b.footprint, bearingDeg, (key / 2) * ratioKm)
    : null;
  b.shadowCache.set(key, poly);
  return poly;
}

export function shadeOn(target, casters, bearingDeg, ratioKm, earlyExitRatio) {
  const pieces = [];
  const byCaster = [];
  let maxRatio = 0;

  for (const other of casters) {
    if (other === target) continue;
    // 건물끼리는 같은 건물(멀티폴리곤)의 다른 조각을 제외한다. 패널에는
    // groupId 가 없고, 호스트 건물은 dh = 0 이라 아래 높이차 조건에서 저절로
    // 빠진다 — 그래서 같은 건물의 더 높은 동이 낮은 동 위의 패널에 드리우는
    // 그림자는 (실제로 생기는 그림자이므로) 그대로 잡힌다.
    if (target.groupId !== undefined && other.groupId === target.groupId) continue;
    // 판정 대상은 지면이 아니라 높이 target.height 의 수평면이다. 상대가 그보다
    // 낮으면 아무리 지면 그림자가 길어도 이 면에는 닿지 않는다.
    const dh = other.height - target.height;
    if (dh < MIN_HEIGHT_DIFF) continue;
    // 이 면까지 닿는 그림자 길이는 전체 높이가 아니라 높이차만큼이다.
    const distKm = dh * ratioKm;
    // 값싼 거리 필터: 그림자가 닿을 수 없을 만큼 멀면 폴리곤 연산까지 가지 않는다
    const gapKm = turf.distance(other.center, target.center, { units: 'kilometers' });
    if (gapKm > distKm + other.radiusKm + target.radiusKm) continue;
    const shadow = shadowAt(other, dh, bearingDeg, ratioKm);
    if (!shadow) continue;
    try {
      // 값싼 예비 검사: 아예 안 겹치면 바로 스킵
      if (!turf.booleanIntersects(target.footprint, shadow)) continue;
      // 겹치긴 하니, 실제로 유의미한 면적인지 확인
      const overlap = turf.intersect(turf.featureCollection([target.footprint, shadow]));
      if (!overlap) continue;
      const ratio = turf.area(overlap) / target.area;
      if (ratio <= 0) continue;
      if (ratio > maxRatio) maxRatio = ratio;
      byCaster.push({ id: other.id, ratio });
      if (earlyExitRatio !== null && ratio >= earlyExitRatio) {
        return { frac: Math.min(1, ratio), exact: true, byCaster };
      }
      if (pieces.length < MAX_OVERLAP_PIECES) pieces.push(overlap);
      if (maxRatio >= 0.999) break;
    } catch (e) { /* 잘못된 지오메트리는 건너뜀 */ }
  }

  if (pieces.length === 0) return { frac: 0, exact: true, byCaster };
  if (pieces.length === 1) {
    return { frac: Math.min(1, turf.area(pieces[0]) / target.area), exact: true, byCaster };
  }
  try {
    // 여러 건물의 그림자를 단순 합산하면 서로 겹치는 부분이 이중 계상된다.
    // 합집합을 한 번 구해 면적을 낸다. (turf.union 은 2개 미만이면 throw 한다)
    const merged = turf.union(turf.featureCollection(pieces));
    return { frac: Math.min(1, turf.area(merged) / target.area), exact: true, byCaster };
  } catch (e) {
    // union 이 실패하면 개별 겹침의 최대값 — 참값의 하한이다
    return { frac: Math.min(1, maxRatio), exact: false, byCaster };
  }
}

export function buildBuildingGrid(buildings) {
  const grid = new Map();
  for (const b of buildings) {
    const [w, s, e, n] = b.bbox;
    for (let gx = Math.floor(w / PANEL_CELL_DEG); gx <= Math.floor(e / PANEL_CELL_DEG); gx++) {
      for (let gy = Math.floor(s / PANEL_CELL_DEG); gy <= Math.floor(n / PANEL_CELL_DEG); gy++) {
        const key = gx + ':' + gy;
        let cell = grid.get(key);
        if (!cell) { cell = []; grid.set(key, cell); }
        cell.push(b);
      }
    }
  }
  return grid;
}

export function findHost(grid, point) {
  const gx = Math.floor(point[0] / PANEL_CELL_DEG);
  const gy = Math.floor(point[1] / PANEL_CELL_DEG);
  const seen = new Set();
  let inside = null;
  let nearest = null;
  let nearestGap = Infinity;

  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (const b of grid.get((gx + dx) + ':' + (gy + dy)) || []) {
        if (seen.has(b)) continue;
        seen.add(b);
        const q = unleanPoint(point, b.height);
        if (turf.booleanPointInPolygon(q, b.footprint)) {
          // 여러 건물에 걸리면 영상에서 위에 보이는 것 = 더 높은 건물을 택한다
          // (밀린 고층 지붕은 앞쪽 저층 건물을 가린다). 높이가 같으면 가장 안쪽
          // (면적이 가장 작은 것)이다.
          if (!inside
            || b.height > inside.height + MIN_HEIGHT_DIFF
            || (Math.abs(b.height - inside.height) <= MIN_HEIGHT_DIFF && b.area < inside.area)) {
            inside = b;
          }
          continue;
        }
        const gap = turf.distance(q, b.center, { units: 'kilometers' }) - b.radiusKm;
        if (gap < nearestGap) { nearestGap = gap; nearest = b; }
      }
    }
  }
  if (inside) return inside;
  // 바닥면 밖이어도 아주 가까우면 그 건물 옥상으로 본다 (시차 보정)
  if (nearest && nearestGap < HOST_SNAP_KM) return nearest;
  return null;
}

// 나주 빛가람동(CENTER) 수평면 전천 일사량, kWh/㎡·일 — NASA POWER 기후값(2001–2025,
// ALLSKY_SFC_SW_DWN). 월별 값이 있어 장마철(7월) 감소까지 월 가중에 들어간다.
export const MONTHLY_GHI = [2.238, 3.077, 4.244, 4.989, 5.443, 4.980, 4.487, 4.639, 4.156, 3.608, 2.499, 1.970];
export const MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
export const monthKwh = (mo) => MONTHLY_GHI[mo - 1] * MONTH_DAYS[mo - 1]; // 그 달 합계(kWh/㎡)
export const ANNUAL_GHI = MONTHLY_GHI.reduce((a, v, i) => a + v * MONTH_DAYS[i], 0); // ≈ 1,410 kWh/㎡·년
export const MODULE_KW = 0.4; // 모듈 한 장(1.0 × 2.0 m) 용량 — 배치 스크립트의 모듈 크기와 맞춘다
export const PERF_RATIO = 0.8; // 인버터·온도·오염 손실
export const MIN_ANNUAL_SUN = 0.7; // 연간 일조율이 이보다 낮은 조각은 설치 대상에서 뺀다
export const ANNUAL_MIN_ALT = 10; // ° — 이보다 낮은 해는 일사가 작고 그림자만 길어 표본에서 뺀다
// 연간 표본은 해가 낮은 겨울 그림자까지 받아야 하므로 순간 분석(100m)보다 여유를 넓힌다
export const ANNUAL_MARGIN_DEG = 0.0025; // ≈ 250m

export function annualSamples() {
  const y = new Date().getFullYear();
  const out = [];
  for (const months of [[12], [1, 11], [2, 10], [3, 9], [4, 8], [5, 7], [6]]) {
    const day = [];
    for (let h = 5; h <= 19; h++) {
      const dt = new Date(Date.UTC(y, months[0] - 1, 21, h - 9, 30));
      const pos = SunCalc.getPosition(dt, CENTER[1], CENTER[0]);
      if (pos.altitude * 180 / Math.PI < ANNUAL_MIN_ALT) continue;
      const s = Math.sin(pos.altitude);
      day.push({
        bearing: (((pos.azimuth * 180 / Math.PI) % 360) + 360) % 360,
        ratioKm: (1 / Math.tan(pos.altitude)) / 1000,
        ghi: 1098 * s * Math.exp(-0.057 / s), // W/㎡
      });
    }
    const dayGhi = day.reduce((a, d) => a + d.ghi, 0);
    const energy = months.reduce((a, mo) => a + monthKwh(mo), 0);
    for (const d of day) out.push({ bearing: d.bearing, ratioKm: d.ratioKm, w: energy * d.ghi / dayGhi });
  }
  return out;
}
