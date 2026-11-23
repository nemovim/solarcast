import { KOREA_BOUNDS, type BuildingCollection, type BuildingInfo } from '../../../vworld-map';

const SIZE = 1000;
const PAGES = 3;
const positive = (value: unknown) => {
  const n = typeof value === 'number' || typeof value === 'string' ? Number(value) : NaN;
  return Number.isFinite(n) && n > 0 ? n : null;
};
const text = (value: unknown) => typeof value === 'string' ? value.trim().slice(0, 120) : '';

export async function GET(request: Request) {
  const bbox = new URL(request.url).searchParams.get('bbox')?.split(',');
  if (!bbox || bbox.length !== 4 || bbox.some((value) => !/^-?\d{1,3}(?:\.\d{1,7})?$/.test(value))) return Response.json({ error: 'invalid-bounds' }, { status: 400 });
  const [w, s, e, n] = bbox.map(Number);
  const [west, south, east, north] = KOREA_BOUNDS;
  if (w < west || s < south || e > east || n > north || e <= w || n <= s || e - w > 0.055 || n - s > 0.055) return Response.json({ error: 'invalid-bounds' }, { status: 400 });

  const key = process.env.VWORLD_KEY?.trim();
  if (!key) return Response.json({ error: 'vworld-not-configured' }, { status: 503 });
  const domain = process.env.VWORLD_DOMAIN?.trim();
  const features: BuildingCollection['features'] = [];
  let partial = false;
  try {
    const signal = AbortSignal.any([request.signal, AbortSignal.timeout(7000)]);
    for (let page = 1; page <= PAGES; page++) {
      const url = new URL('https://api.vworld.kr/req/data');
      url.search = new URLSearchParams({ key, ...(domain ? { domain } : {}), service: 'data', version: '2.0', request: 'GetFeature', data: 'LT_C_SPBD', format: 'json', geometry: 'true', attribute: 'true', crs: 'EPSG:4326', geomFilter: `BOX(${w},${s},${e},${n})`, size: String(SIZE), page: String(page) }).toString();
      const upstream = await fetch(url, { cache: 'no-store', redirect: 'error', signal });
      if (!upstream.ok) throw new Error('upstream error');
      const { response } = await upstream.json();
      if (response?.status === 'NOT_FOUND') break;
      const items = response?.result?.featureCollection?.features;
      if (response?.status !== 'OK' || !Array.isArray(items)) throw new Error('upstream payload');
      for (const feature of items.slice(0, SIZE)) {
        const geometry = feature?.geometry;
        if (!['Polygon', 'MultiPolygon'].includes(geometry?.type) || !Array.isArray(geometry.coordinates)) continue;
        const p = feature.properties ?? {};
        const height = positive(p.heit ?? p.height);
        const floors = positive(p.gro_flo_co);
        const properties: BuildingInfo = { name: text(p.buld_nm) || text(p.buld_nm_dc) || '건물', heightM: Math.min(1000, height ?? (floors ? floors * 3.3 : 15)), heightSource: height ? 'record' : floors ? 'floors' : 'default' };
        features.push({ type: 'Feature', geometry, properties });
      }
      const total = positive(response.page?.total);
      if (total !== null ? page >= total : items.length < SIZE) break;
      if (page === PAGES) partial = true;
    }
    return Response.json({ type: 'FeatureCollection', features, partial }, { headers: { 'Cache-Control': 'private, no-store' } });
  } catch {
    return Response.json({ error: 'vworld-buildings-unavailable' }, { status: 502 });
  }
}
