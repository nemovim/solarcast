export const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
export function shuffledItems(items, seed) {
  // Fixed seed keeps the random order stable across reloads and preserves assignments.
  let value=2166136261;
  for(const char of seed) value=Math.imul(value^char.charCodeAt(0),16777619)>>>0;
  const result=[...items];
  for(let i=result.length-1;i>0;i--) {
    value=(value+0x6D2B79F5)>>>0;
    let random=Math.imul(value^(value>>>15),value|1);
    random^=random+Math.imul(random^(random>>>7),random|61);
    const j=Math.floor(((random^(random>>>14))>>>0)/4294967296*(i+1));
    [result[i],result[j]]=[result[j],result[i]];
  }
  return result;
}
export function rectangleFromDiagonal(a, b) {
  const left=Math.min(a[0],b[0]),right=Math.max(a[0],b[0]);
  const top=Math.min(a[1],b[1]),bottom=Math.max(a[1],b[1]);
  return [[left,top],[right,top],[right,bottom],[left,bottom]];
}
export function polygonError(points) {
  if (points.length < 3) return '꼭짓점을 3개 이상 찍어 주세요.';
  if (new Set(points.map(p => p.join(','))).size !== points.length) return '같은 위치를 두 번 찍을 수 없습니다.';
  const cross = (a,b,c) => (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const on = (a,b,c) => Math.abs(cross(a,b,c)) < 1e-8 && c[0]>=Math.min(a[0],b[0]) && c[0]<=Math.max(a[0],b[0]) && c[1]>=Math.min(a[1],b[1]) && c[1]<=Math.max(a[1],b[1]);
  const n=points.length;
  const area=Math.abs(points.reduce((s,p,i)=>s+p[0]*points[(i+1)%n][1]-points[(i+1)%n][0]*p[1],0))/2;
  if(area<0.5) return '영역이 너무 작거나 점 순서가 잘못되었습니다.';
  for(let i=0;i<n;i++) for(let j=i+1;j<n;j++) {
    if(j===i+1 || (i===0 && j===n-1)) continue;
    const a=points[i],b=points[(i+1)%n],c=points[j],d=points[(j+1)%n];
    if((cross(a,b,c)*cross(a,b,d)<0 && cross(c,d,a)*cross(c,d,b)<0)||on(a,b,c)||on(a,b,d)||on(c,d,a)||on(c,d,b)) return '변이 교차합니다. 패널 경계를 따라 순서대로 찍어 주세요.';
  }
  return '';
}
export function translatePolygon(points, dx, dy, width, height) {
  dx=clamp(dx,-Math.min(...points.map(p=>p[0])),width-Math.max(...points.map(p=>p[0])));
  dy=clamp(dy,-Math.min(...points.map(p=>p[1])),height-Math.max(...points.map(p=>p[1])));
  return points.map(([x,y])=>[x+dx,y+dy]);
}
