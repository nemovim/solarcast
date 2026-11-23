import { KOREA_BOUNDS } from '../../../../../../vworld-map';

export async function GET(_request: Request, { params }: { params: Promise<{ z: string; x: string; y: string }> }) {
  const tile = await params;
  if (![tile.z, tile.x, tile.y].every((value) => /^\d{1,6}$/.test(value))) return new Response(null, { status: 400 });
  const [z, x, y] = [Number(tile.z), Number(tile.x), Number(tile.y)];
  const n = 2 ** z;
  if (z < 5 || z > 19 || x >= n || y >= n) return new Response(null, { status: 400 });
  const lat = (row: number) => Math.atan(Math.sinh(Math.PI * (1 - 2 * row / n))) * 180 / Math.PI;
  const [west, south, east, north] = KOREA_BOUNDS;
  if ((x + 1) / n * 360 - 180 < west || x / n * 360 - 180 > east || lat(y) < south || lat(y + 1) > north) return new Response(null, { status: 404 });

  const key = process.env.VWORLD_KEY?.trim();
  if (!key) return new Response(null, { status: 503 });
  try {
    const domain = process.env.VWORLD_DOMAIN?.trim();
    const url = new URL(`https://api.vworld.kr/req/wmts/1.0.0/${encodeURIComponent(key)}/Satellite/${z}/${y}/${x}.jpeg`);
    const response = await fetch(url, { cache: 'no-store', redirect: 'error', signal: AbortSignal.timeout(5500), headers: domain ? { Referer: domain } : {} });
    const type = response.headers.get('content-type')?.split(';')[0];
    if (response.status === 404) return new Response(null, { status: 404 });
    if (!response.ok || !response.body || !type || !['image/jpeg', 'image/png', 'image/webp'].includes(type)) return new Response(null, { status: 502 });
    return new Response(response.body, { headers: { 'Content-Type': type, 'Cache-Control': 'private, no-store' } });
  } catch {
    return new Response(null, { status: 502 });
  }
}
