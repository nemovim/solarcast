import * as turf from '@turf/turf';
import * as SunCalc from 'suncalc';
import type { Feature, FeatureCollection, MultiPolygon, Polygon } from 'geojson';
import type { Viewport } from './map-panel';
import type { AnalysisResult, Summary } from './analysis';
import type { BuildingInfo, BuildingCollection } from './vworld-map';
import { extent } from './client-imagery';
import type { PanelFeature, RoofFeature } from './client-precomputed';

type PolygonFeature = Feature<Polygon | MultiPolygon>;
type Building = { geometry: PolygonFeature; bbox: number[]; height: number; name: string; id: string; source: string };
const empty = (): FeatureCollection<Polygon | MultiPolygon> => ({ type:'FeatureCollection', features:[] });
const boxesOverlap = (a: number[], b: number[]) => a[0] <= b[2] && a[2] >= b[0] && a[1] <= b[3] && a[3] >= b[1];
function intersection(a: PolygonFeature, b: PolygonFeature): PolygonFeature | null {
  if (!turf.booleanIntersects(a,b)) return null;
  return turf.intersect(turf.featureCollection([a,b])) as PolygonFeature | null;
}

function direction(geometry: PolygonFeature, lat: number): number {
  const ring = geometry.geometry.type === 'Polygon' ? geometry.geometry.coordinates[0] : geometry.geometry.coordinates[0][0];
  let longest = 0, bearing = 0;
  for (let i=1; i<ring.length; i++) {
    const dx=(ring[i][0]-ring[i-1][0])*Math.cos(lat*Math.PI/180), dy=ring[i][1]-ring[i-1][1];
    const length=dx*dx+dy*dy;
    if (length>longest) { longest=length; bearing=Math.atan2(dx,dy)*180/Math.PI; }
  }
  return bearing;
}

function rings(geometry: PolygonFeature): number[][][] {
  return geometry.geometry.type === 'Polygon' ? geometry.geometry.coordinates : geometry.geometry.coordinates.flat();
}

// Roof geometry is evaluated on a 0.125 m grid aligned with the building's longest wall. The
// roof masks are traced from ~0.3 m image pixels, so the grid keeps their detail while replacing
// exact polygon clipping and buffering. On a 48-tile sample it gave about 2.6% less usable area
// and 1.5% more modules than exact polygon operations.
const CELL=0.125, SETBACK=1;
// 1 m x 2 m modules in rows 3 m apart, tried at 0.25 m x 0.75 m offsets (16 layouts), in cells.
const MODULE_U=1/CELL, MODULE_V=2/CELL, ROW_PITCH=3/CELL, STEP_U=.25/CELL, STEP_V=.75/CELL;
const PAD=Math.ceil(SETBACK/CELL)+1;

// Building frame: (u, v) metres rotated so the longest wall points along +v.
type Frame = { cx: number; cy: number; cos: number; sin: number; u0: number; v0: number; w: number; h: number };
const toFrame=(f: Frame, x: number, y: number) => [(x-f.cx)*f.cos-(y-f.cy)*f.sin, (x-f.cx)*f.sin+(y-f.cy)*f.cos];
const fromFrame=(f: Frame, u: number, v: number) => [f.cx+u*f.cos+v*f.sin, f.cy-u*f.sin+v*f.cos];

// Fills cells whose centers fall inside any of the polygons (even-odd within each polygon,
// so holes stay empty). Polygons are lists of rings in frame coordinates.
function rasterize(f: Frame, polygons: number[][][][], out=new Uint8Array(f.w*f.h)): Uint8Array {
  const crossings: number[]=[];
  for (const polygon of polygons) {
    let low=Infinity, high=-Infinity;
    for (const ring of polygon) for (const p of ring) { low=Math.min(low,p[1]); high=Math.max(high,p[1]); }
    const j0=Math.max(0,Math.ceil((low-f.v0)/CELL-.5)), j1=Math.min(f.h-1,Math.floor((high-f.v0)/CELL-.5));
    for (let j=j0;j<=j1;j++) {
      const v=f.v0+(j+.5)*CELL;
      crossings.length=0;
      for (const ring of polygon) for (let k=1;k<ring.length;k++) {
        const a=ring[k-1], b=ring[k];
        if ((a[1]>v) !== (b[1]>v)) crossings.push(a[0]+(v-a[1])*(b[0]-a[0])/(b[1]-a[1]));
      }
      crossings.sort((a,b)=>a-b);
      for (let k=0;k+1<crossings.length;k+=2) {
        const i0=Math.max(0,Math.ceil((crossings[k]-f.u0)/CELL-.5)), i1=Math.min(f.w-1,Math.floor((crossings[k+1]-f.u0)/CELL-.5));
        for (let i=i0;i<=i1;i++) out[j*f.w+i]=1;
      }
    }
  }
  return out;
}

// Counts filled cells in any rectangle in constant time (summed-area table).
function areaCounter(mask: Uint8Array, w: number, h: number) {
  const sat=new Int32Array((w+1)*(h+1));
  for (let j=0;j<h;j++) for (let i=0,row=0;i<w;i++) { row+=mask[j*w+i]; sat[(j+1)*(w+1)+i+1]=sat[j*(w+1)+i+1]+row; }
  // Filled cells in columns [i0, i1) and rows [j0, j1).
  return (i0: number, j0: number, i1: number, j1: number) => sat[j1*(w+1)+i1]-sat[j0*(w+1)+i1]-sat[j1*(w+1)+i0]+sat[j0*(w+1)+i0];
}

// Keeps cells with no empty cell within SETBACK along either grid axis, so each kept cell lies
// wholly beyond the setback. Square neighborhoods trim slightly more than a round buffer at inner
// corners and walls that are not aligned to the grid.
function setback(f: Frame, mask: Uint8Array): Uint8Array {
  const r=Math.round(SETBACK/CELL), count=areaCounter(mask,f.w,f.h), out=new Uint8Array(mask.length), full=(2*r+1)**2;
  for (let j=r;j<f.h-r;j++) for (let i=r;i<f.w-r;i++) if (mask[j*f.w+i] && count(i-r,j-r,i+r+1,j+r+1)===full) out[j*f.w+i]=1;
  return out;
}

// Picks the module layout (rows of at least three modules) with the most modules among the grid
// offsets. Returns module centers (u, v).
function layout(f: Frame, usable: Uint8Array): number[][] {
  const count=areaCounter(usable,f.w,f.h);
  let best: number[][]=[];
  for (let py=0;py<ROW_PITCH;py+=STEP_V) for (let px=0;px<MODULE_U;px+=STEP_U) {
    const current: number[][]=[];
    for (let j=py;j+MODULE_V<=f.h;j+=ROW_PITCH) {
      let run: number[][]=[];
      for (let i=px;i<=f.w;i+=MODULE_U) {
        if (i+MODULE_U<=f.w && count(i,j,i+MODULE_U,j+MODULE_V)===MODULE_U*MODULE_V) { run.push([f.u0+(i+MODULE_U/2)*CELL,f.v0+(j+MODULE_V/2)*CELL]); continue; }
        if (run.length>=3) current.push(...run);
        run=[];
      }
    }
    if (current.length>best.length) best=current;
  }
  return best;
}

// Merges filled cells into rectangles (row runs extended downward) for display.
function rectangles(f: Frame, mask: Uint8Array, toLonLat: (x: number, y: number)=>number[]): MultiPolygon {
  const polygons: number[][][][]=[];
  const corner=(u: number, v: number) => { const [x,y]=fromFrame(f,u,v); return toLonLat(x,y); };
  let open=new Map<number, [number, number]>(); // run start -> [run end, first row]
  for (let j=0;j<=f.h;j++) {
    const next=new Map<number, [number, number]>();
    if (j<f.h) for (let i=0;i<f.w;) {
      if (!mask[j*f.w+i]) { i++; continue; }
      const start=i; while (i<f.w && mask[j*f.w+i]) i++;
      const previous=open.get(start);
      next.set(start,[i,previous && previous[0]===i ? previous[1] : j]);
    }
    for (const [start,[end,top]] of open) {
      const continued=next.get(start);
      if (continued && continued[0]===end && continued[1]===top) continue;
      const u0=f.u0+start*CELL, u1=f.u0+end*CELL, v0=f.v0+top*CELL, v1=f.v0+j*CELL;
      polygons.push([[corner(u0,v0),corner(u1,v0),corner(u1,v1),corner(u0,v1),corner(u0,v0)]]);
    }
    open=next;
  }
  return {type:'MultiPolygon',coordinates:polygons};
}

// Shading uses a horizon profile instead of per-hour shadow polygons. From a module center,
// each azimuth bin keeps the steepest obstruction tan(elevation) = rise / distance over taller
// nearby footprints. The sun is visible at (altitude, azimuth) when tan(altitude) exceeds that
// bin's horizon, so the whole year reduces to one table lookup per bin. Diffuse light uses an
// isotropic sky reduced by the horizon (sky view factor). PVGIS applies horizons the same way.
const AZIMUTH_BINS=120, BIN_DEGREES=360/AZIMUTH_BINS;
const SAMPLE_DAYS=[5,15,25], SAMPLE_MINUTES=10;
type SkyBin = { tan: number[]; above: number[] };
// Annual horizontal irradiation (kWh/m2): direct binned by sun azimuth, total diffuse and global.
export type SkyModel = { bins: SkyBin[]; direct: number; diffuse: number; global: number };
type Caster = { rings: number[][][]; rise: number };

// Spreads each month's direct horizontal irradiation over that month's sun positions in
// proportion to a clear-sky direct horizontal shape, sin(a) * exp(-0.057 / sin(a)).
export function buildSky(lat: number, lon: number, monthly: number[], diffuse: number[]): SkyModel {
  const raw=Array.from({length:AZIMUTH_BINS},()=>[] as number[][]);
  for (let month=0;month<12;month++) {
    const samples: number[][]=[];
    let total=0;
    for (const day of SAMPLE_DAYS) for (let minute=SAMPLE_MINUTES/2;minute<1440;minute+=SAMPLE_MINUTES) {
      // suncalc 2.x returns degrees: altitude above the horizon and azimuth as a bearing from
      // north. Its bundled @types (1.x) still describe the old radian, south-based values.
      const sun=SunCalc.getPosition(new Date(Date.UTC(2026,month,day,-9,minute)),lat,lon);
      if (sun.altitude<=0) continue;
      const altitude=sun.altitude*Math.PI/180, sine=Math.sin(altitude), weight=sine*Math.exp(-.057/sine);
      samples.push([Math.round(sun.azimuth/BIN_DEGREES)%AZIMUTH_BINS,Math.tan(altitude),weight]);
      total+=weight;
    }
    const scale=total>0 ? (monthly[month]-diffuse[month])/total : 0;
    for (const [bin,tangent,weight] of samples) raw[bin].push([tangent,weight*scale]);
  }
  const bins=raw.map((items) => {
    items.sort((a,b)=>a[0]-b[0]);
    const above=Array<number>(items.length);
    for (let i=items.length-1, sum=0;i>=0;i--) above[i]=sum+=items[i][1];
    return {tan:items.map((item)=>item[0]),above};
  });
  const direct=monthly.reduce((sum,v,i)=>sum+v-diffuse[i],0), diffuseTotal=diffuse.reduce((a,b)=>a+b,0);
  return {bins,direct,diffuse:diffuseTotal,global:direct+diffuseTotal};
}

// Steepest obstruction per azimuth bin, as tan(elevation), seen from local point (x, y) in metres.
export function horizonProfile(x: number, y: number, casters: Caster[]): Float64Array {
  const horizon=new Float64Array(AZIMUTH_BINS);
  for (const {rings: parts,rise} of casters) for (const ring of parts) for (let i=1;i<ring.length;i++) {
    const ax=ring[i-1][0]-x, ay=ring[i-1][1]-y, ex=ring[i][0]-ring[i-1][0], ey=ring[i][1]-ring[i-1][1];
    const start=Math.atan2(ax,ay)*180/Math.PI;
    const span=((Math.atan2(ax+ex,ay+ey)*180/Math.PI-start+540)%360)-180;
    for (let k=Math.ceil(Math.min(start,start+span)/BIN_DEGREES);k<=Math.floor(Math.max(start,start+span)/BIN_DEGREES);k++) {
      const angle=k*BIN_DEGREES*Math.PI/180, ux=Math.sin(angle), uy=Math.cos(angle);
      const denominator=ux*ey-uy*ex;
      if (Math.abs(denominator)<1e-12) continue;
      // Distance along the ray to the edge: t = (a x e) / (u x e).
      const distance=(ax*ey-ay*ex)/denominator;
      if (distance<=0) continue;
      const bin=((k%AZIMUTH_BINS)+AZIMUTH_BINS)%AZIMUTH_BINS;
      horizon[bin]=Math.max(horizon[bin],rise/distance);
    }
  }
  return horizon;
}

// Annual horizontal irradiation (kWh/m2) reaching a point with the given horizon profile.
export function annualYield(sky: SkyModel, horizon: Float64Array): number {
  let direct=0, view=0;
  for (let k=0;k<AZIMUTH_BINS;k++) {
    const h=horizon[k], {tan,above}=sky.bins[k];
    view+=1/(1+h*h); // cos^2 of the horizon elevation
    let low=0, high=tan.length;
    while (low<high) { const mid=(low+high)>>1; if (tan[mid]>h) high=mid; else low=mid+1; }
    if (low<tan.length) direct+=above[low];
  }
  return direct+sky.diffuse*view/AZIMUTH_BINS;
}

function convexHull(points: number[][]): number[][] {
  const sorted=[...points].sort((a,b)=>a[0]-b[0] || a[1]-b[1]);
  const cross=(o: number[], a: number[], b: number[]) => (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
  const half=(list: number[][]) => {
    const hull: number[][]=[];
    for (const point of list) {
      while (hull.length>=2 && cross(hull[hull.length-2],hull[hull.length-1],point)<=0) hull.pop();
      hull.push(point);
    }
    hull.pop();
    return hull;
  };
  const hull=[...half(sorted),...half([...sorted].reverse())];
  return [...hull,hull[0]];
}

export async function estimateSolar(viewport: Viewport, panels: PanelFeature[], roofs: RoofFeature[], progress: (s:string)=>void, coverage?: FeatureCollection<Polygon>): Promise<AnalysisResult> {
  extent(viewport); // Validate the entire viewport before working with its intersection.
  const closed=[...viewport,viewport[0]];
  const viewportPolygon=turf.polygon([closed]) as PolygonFeature;
  const mergedCoverage = coverage ? (coverage.features.length > 1 ? turf.union(coverage) as PolygonFeature | null : coverage.features[0] ?? null) : null;
  const roi = coverage ? (mergedCoverage ? intersection(viewportPolygon, mergedCoverage) : null) : viewportPolygon;
  if (!roi) throw new Error('화면에 분석 가능한 사전 추론 영역이 없습니다.');
  const [west,south,east,north] = turf.bbox(roi);
  const lat=(south+north)/2, lon=(west+east)/2;
  const yMargin=300/110574, xMargin=300/(111320*Math.cos(lat*Math.PI/180));
  const bbox=[Math.max(123,west-xMargin),Math.max(32,south-yMargin),Math.min(133,east+xMargin),Math.min(40,north+yMargin)].map((n)=>n.toFixed(6)).join(',');
  progress('건물과 일사량을 불러오는 중입니다.');
  const [buildingResponse, irradianceResponse]=await Promise.all([
    fetch(`/api/vworld/buildings?bbox=${bbox}`),fetch(`/api/irradiance?lat=${lat}&lon=${lon}`),
  ]);
  if (!buildingResponse.ok) throw new Error('건물 정보를 불러오지 못했습니다.');
  const data: BuildingCollection & { partial?: boolean }=await buildingResponse.json();
  if (data.partial) throw new Error('건물 정보가 너무 많습니다. 지도를 확대해 주세요.');
  const irradiance = irradianceResponse.ok ? await irradianceResponse.json() : null;
  const monthly: number[] | null = Array.isArray(irradiance?.monthly) && Array.isArray(irradiance?.diffuse) && irradiance.monthly.length===12 && irradiance.diffuse.length===12 && irradiance.monthly.every((v: unknown,i: number)=>typeof v==='number' && Number.isFinite(v) && v>0 && typeof irradiance.diffuse[i]==='number' && Number.isFinite(irradiance.diffuse[i]) && irradiance.diffuse[i]>=0 && irradiance.diffuse[i]<=v) ? irradiance.monthly : null;
  const diffuse: number[] | null = monthly ? irradiance.diffuse : null;
  const sky = monthly && diffuse ? buildSky(lat,lon,monthly,diffuse) : null;
  const kx=111320*Math.cos(lat*Math.PI/180), ky=110574;
  const local=(point: number[]) => [(point[0]-lon)*kx,(point[1]-lat)*ky];
  const mapped: Building[]=data.features.map((ft,i) => ({geometry:ft as PolygonFeature,bbox:turf.bbox(ft),height:ft.properties.heightM,name:ft.properties.name,id:`building-${i}`,source:ft.properties.heightSource}));
  const bounds=mapped.map((item)=>item.bbox);
  // Casters use the convex hull of their outline: it can only bring an obstruction closer, and it
// keeps detailed outlines from multiplying the horizon cost.
  const localRings=new Map(mapped.map((item)=>[item,[convexHull(rings(item.geometry)[0].map(local))]]));
  // Buildings whose bounding boxes come within 300 m can shade each other.
  const nearby=mapped.map((a) => {
    const [w,s,e,n]=a.bbox, mx=300/kx, my=300/ky, box=[w-mx,s-my,e+mx,n+my];
    return mapped.filter((b,j) => a!==b && boxesOverlap(bounds[j],box));
  });
  const results: AnalysisResult={ buildings:empty(),existing_panels:empty(),available_roofs:empty(),summaries:[],warnings:[],totals:{availableAreaM2:null,panelAreaM2:null,capacityKw:null,annualEnergyKwh:null},excluded:{availableAreaM2:0,panelAreaM2:0,capacityKw:0,annualEnergyKwh:0},models:{roof:null,panel:'NLR RTMDet-Ins'} };
  const panelGeometries=panels.map((panel) => { const clip=intersection(panel,roi); return clip ? {...clip,properties:panel.properties} : null; }).filter((p):p is NonNullable<typeof p> => p!==null);
  results.existing_panels.features = panelGeometries;
  if (!monthly) results.warnings.push('일사량 조회 실패: 발전량은 미계산입니다.');
  results.warnings.push('SAM 옥상 마스크와 건물 외곽선의 겹치는 부분을 평평한 옥상으로 가정하며 옥탑·설비·나무·경사는 반영하지 않습니다.');
  results.warnings.push('각 모듈 중심에서 300m 안의 더 높은 건물로 3° 간격 방위별 지평선을 구합니다. 월별 직달 일사는 매월 3일의 10분 간격 태양 위치에 맑은 날 분포로 나누어 지평선 위의 몫만 더합니다. 산란 일사는 등방성 하늘에서 지평선이 가리는 만큼 줄입니다.');
  results.warnings.push('수평 설치, 성능계수 0.8을 사용하며 연간 일사량이 가리지 않은 수평면의 70% 이상인 모듈만 포함합니다. 기준은 임의 가정입니다. 패널 경사, 전기적 부분 음영 손실, 반사광은 반영하지 않아 실제 발전량과 차이가 날 수 있습니다.');
  const assumed=mapped.filter((b)=>b.source==='default').length;
  const fromFloors=mapped.filter((b)=>b.source==='floors').length;
  if (assumed) results.warnings.push(`높이가 없는 건물 ${assumed}개는 15m로 가정했습니다. 그림자와 발전량의 불확실성이 큽니다.`);
  if (fromFloors) results.warnings.push(`건물 ${fromFloors}개 높이는 지상 층수 × 3.3m로 추정했습니다.`);
  // Roof tiles are one MultiPolygon per tile; splitting them lets each building rasterize only
  // the nearby parts.
  const roofParts=roofs.flatMap((roof) => roof.geometry.type === 'Polygon' ? [roof.geometry.coordinates] : roof.geometry.coordinates);
  const roofBounds=roofParts.map((roof)=>turf.bbox(turf.polygon(roof)));
  const panelParts=panelGeometries.flatMap((panel) => panel.geometry.type === 'Polygon' ? [panel.geometry.coordinates] : panel.geometry.coordinates);
  const panelBounds=panelParts.map((panel)=>turf.bbox(turf.polygon(panel)));
  const roiParts=roi.geometry.type === 'Polygon' ? [roi.geometry.coordinates] : roi.geometry.coordinates;
  const roiBounds=turf.bbox(roi);
  const toLonLat=(x: number, y: number) => [lon+x/kx, lat+y/ky];
  let missingRoof=0, lastYield=performance.now();
  for (let index=0; index<mapped.length; index++) {
    const building=mapped[index];
    if (!boxesOverlap(building.bbox,roiBounds)) continue;
    // Building frame and grid, padded so the setback also applies at the grid edge.
    const [cx,cy]=local([(building.bbox[0]+building.bbox[2])/2,(building.bbox[1]+building.bbox[3])/2]);
    const angle=direction(building.geometry,lat)*Math.PI/180;
    const frame: Frame={cx,cy,cos:Math.cos(angle),sin:Math.sin(angle),u0:0,v0:0,w:0,h:0};
    const toUV=(polygon: number[][][]) => polygon.map((ring)=>ring.map((point)=>{ const [x,y]=local(point); return toFrame(frame,x,y); }));
    const footprintParts=(building.geometry.geometry.type === 'Polygon' ? [building.geometry.geometry.coordinates] : building.geometry.geometry.coordinates).map(toUV);
    let uMin=Infinity, uMax=-Infinity, vMin=Infinity, vMax=-Infinity;
    for (const part of footprintParts) for (const [u,v] of part[0]) { uMin=Math.min(uMin,u); uMax=Math.max(uMax,u); vMin=Math.min(vMin,v); vMax=Math.max(vMax,v); }
    frame.u0=uMin-PAD*CELL; frame.v0=vMin-PAD*CELL;
    frame.w=Math.ceil((uMax-uMin)/CELL)+2*PAD; frame.h=Math.ceil((vMax-vMin)/CELL)+2*PAD;
    if (frame.w*frame.h > 8_000_000) throw new Error('건물 면적이 계산 가능한 범위를 넘었습니다.');
    const inFootprint=rasterize(frame,footprintParts);
    const inRoi=rasterize(frame,roiParts.map(toUV));
    let footprintCells=0, roiCells=0;
    for (let i=0;i<inFootprint.length;i++) { footprintCells+=inFootprint[i]; inFootprint[i]&=inRoi[i]; roiCells+=inFootprint[i]; }
    if (!roiCells) continue;
    // Clip the displayed outline only when the analysis area actually cuts it.
    const footprint=roiCells===footprintCells ? building.geometry : intersection(building.geometry,roi) ?? building.geometry;
    const near=(bounds: number[]) => boxesOverlap(bounds,building.bbox);
    const usable=rasterize(frame,roofParts.filter((_,i)=>near(roofBounds[i])).map(toUV));
    let roofCells=0;
    for (let i=0;i<usable.length;i++) { usable[i]&=inFootprint[i]; roofCells+=usable[i]; }
    if (!roofCells) {
      missingRoof++;
      const summary: Summary={buildingId:building.id,name:building.name,availableAreaM2:null,panelAreaM2:null,capacityKw:null,annualEnergyKwh:null,heightSource:building.source};
      results.summaries.push(summary);
      results.buildings.features.push({...footprint,properties:summary});
      continue;
    }
    const existing=rasterize(frame,panelParts.filter((_,i)=>near(panelBounds[i])).map(toUV));
    for (let i=0;i<usable.length;i++) usable[i]&=1-existing[i];
    const kept=setback(frame,usable);
    let usableCells=0;
    for (let i=0;i<kept.length;i++) usableCells+=kept[i];
    const area=usableCells*CELL*CELL;
    const centers=usableCells ? layout(frame,kept) : [];
    let capacity: number | null=null, energy: number | null=null, panelArea: number | null=null;
    if (usableCells) results.available_roofs.features.push({type:'Feature',geometry:rectangles(frame,kept,toLonLat),properties:{buildingId:building.id,areaM2:Math.round(area*100)/100}});
    if (sky) {
      const casters: Caster[]=nearby[index].filter((item)=>item.height>building.height)
        .map((item)=>({rings:localRings.get(item)!,rise:item.height-building.height}));
      let moduleCount=0, irradiation=0;
      for (const [u,v] of centers) {
        const [x,y]=fromFrame(frame,u,v), annual=annualYield(sky,horizonProfile(x,y,casters));
        if (annual/sky.global>=.7) { moduleCount++; irradiation+=annual; }
      }
      panelArea=moduleCount*2;
      capacity=.4*moduleCount;
      energy=.4*.8*irradiation;
    }
    const summary: Summary={ buildingId:building.id,name:building.name,availableAreaM2:Math.round(area*100)/100,panelAreaM2:panelArea,capacityKw:capacity===null?null:Math.round(capacity*100)/100,annualEnergyKwh:energy===null?null:Math.round(energy*10)/10,heightSource:building.source };
    results.summaries.push(summary);
    results.buildings.features.push({...footprint,properties:summary});
    // Yield to the page about every 50 ms rather than per building (timers clamp to ~4 ms).
    if (performance.now()-lastYield>50) {
      progress(`건물 면적·그림자 계산 중 (${results.summaries.length})`);
      await new Promise((resolve)=>setTimeout(resolve,0));
      lastYield=performance.now();
    }
  }
  if (missingRoof) results.warnings.push(`건물 ${missingRoof}개는 선택한 모델의 옥상 탐지 결과가 없어 계산하지 못했습니다.`);
  // Totals cover only buildings with a computed value; each total reports how many it excludes.
  for (const key of ['availableAreaM2','panelAreaM2','capacityKw','annualEnergyKwh'] as const) {
    const values=results.summaries.map((item)=>item[key]).filter((v): v is number => v !== null);
    results.totals[key]=values.length ? Math.round(values.reduce((sum,v)=>sum+v,0)*100)/100 : null;
    results.excluded[key]=results.summaries.length-values.length;
  }
  return results;
}
