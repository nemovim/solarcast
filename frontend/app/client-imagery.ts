import type { Viewport } from './map-panel';

const ZOOM = 19;
const SIZE = 256;
const SCALE = 2 ** ZOOM * SIZE;
const KOREA = { west: 123, south: 32, east: 133, north: 40 };

export function worldPixel(lng: number, lat: number): [number, number] {
  const radians = lat * Math.PI / 180;
  return [(lng + 180) / 360 * SCALE, (1 - Math.asinh(Math.tan(radians)) / Math.PI) / 2 * SCALE];
}

export function worldCoordinate(x: number, y: number): [number, number] {
  return [x / SCALE * 360 - 180, Math.atan(Math.sinh(Math.PI * (1 - 2 * y / SCALE))) * 180 / Math.PI];
}

export function extent(viewport: Viewport) {
  const lon = viewport.map((p) => p[0]), lat = viewport.map((p) => p[1]);
  const [west, south, east, north] = [Math.min(...lon), Math.min(...lat), Math.max(...lon), Math.max(...lat)];
  if (west < KOREA.west || east > KOREA.east || south < KOREA.south || north > KOREA.north || west >= east || south >= north) throw new Error('국내 지도 범위를 선택해 주세요.');
  return { west, south, east, north };
}

