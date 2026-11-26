import * as turf from '@turf/turf';
import * as SunCalc from 'suncalc';
import type { Feature, FeatureCollection, MultiPolygon, Polygon } from 'geojson';
import type { Viewport } from './map-panel';
import type { AnalysisResult, Summary } from './analysis';
import type { BuildingInfo, BuildingCollection } from './vworld-map';
import { extent } from './client-imagery';
import type { PanelFeature, RoofFeature } from './client-precomputed';

type PolygonFeature = Feature<Polygon | MultiPolygon>;
type Building = { geometry: PolygonFeature; height: number | null; name: string; id: string; source: string };
type ModuleRow = { geometry: PolygonFeature; count: number };
const empty = (): FeatureCollection<Polygon | MultiPolygon> => ({ type:'FeatureCollection', features:[] });
const boxesOverlap = (a: number[], b: number[]) => a[0] <= b[2] && a[2] >= b[0] && a[1] <= b[3] && a[3] >= b[1];
// Fixed representative dates approximate the equinoxes and solstices.
const seasonalDates = [[2,20],[5,21],[8,23],[11,22]] as const;
const seasonalDays = seasonalDates.map(([month,day]) => Date.UTC(2026,month,day));
const yearMs = 365 * 24 * 60 * 60 * 1000;

export function monthlyShadeFactors(seasonal: number[]): number[] {
  return Array.from({length:12}, (_, month) => {
    let date=Date.UTC(2026,month,15);
    if (date < seasonalDays[0]) date += yearMs;
    const index=seasonalDays.findLastIndex((day)=>day<=date);
    const next=(index+1)%4;
    const end=next===0 ? seasonalDays[0]+yearMs : seasonalDays[next];
    const fraction=(date-seasonalDays[index])/(end-seasonalDays[index]);
    return seasonal[index]*(1-fraction)+seasonal[next]*fraction;
  });
}

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

function modules(candidate: PolygonFeature, roof: PolygonFeature): ModuleRow[] {
  const center=turf.centroid(roof).geometry.coordinates;
  const angle=direction(roof,center[1]);
  const aligned=turf.transformRotate(candidate,-angle,{pivot:center}) as PolygonFeature;
  const [minX,minY,maxX,maxY]=turf.bbox(aligned);
  const dx=1/(111320*Math.cos(center[1]*Math.PI/180)), dy=1/110574;
  if ((maxX-minX)/dx*(maxY-minY)/dy > 2_000_000) throw new Error('건물 면적이 계산 가능한 범위를 넘었습니다.');
  let best: ModuleRow[]=[], bestCount=0;
  for (let py=0;py<4;py++) for (let px=0;px<4;px++) {
    const current: ModuleRow[]=[];
    for (let y=minY+py*.75*dy; y+2*dy<=maxY; y+=3*dy) {
      let run: PolygonFeature[]=[];
      for (let x=minX+px*.25*dx; x<=maxX; x+=dx) {
        const cell=turf.bboxPolygon([x,y,x+dx,y+2*dy]) as PolygonFeature;
        if (x+dx<=maxX && turf.booleanContains(aligned,cell)) run.push(cell);
        else { if (run.length>=3) { const first=turf.bbox(run[0]), last=turf.bbox(run[run.length-1]); current.push({geometry:turf.bboxPolygon([first[0],first[1],last[2],last[3]]) as PolygonFeature,count:run.length}); } run=[]; }
      }
      if (run.length>=3) { const first=turf.bbox(run[0]), last=turf.bbox(run[run.length-1]); current.push({geometry:turf.bboxPolygon([first[0],first[1],last[2],last[3]]) as PolygonFeature,count:run.length}); }
    }
    const count=current.reduce((sum,row)=>sum+row.count,0);
    if (count>bestCount) { best=current; bestCount=count; }
  }
  return best.map((row) => ({...row,geometry:turf.transformRotate(row.geometry,angle,{pivot:center}) as PolygonFeature}));
}

// Monthly horizontal diffuse remains unshaded; direct horizontal is scaled by the
// representative-day sunlit fraction. Sky obstruction and electrical mismatch are not modeled.
export function annualRowYield(monthly: number[], diffuse: number[], directSunlit: number[]): number {
  return monthly.reduce((sum,ghi,i)=>sum+diffuse[i]+(ghi-diffuse[i])*directSunlit[i],0);
}

function shadow(caster: Building, target: Building, altitude: number, bearing: number): PolygonFeature | null {
  if (caster.height === null || target.height === null || caster.height <= target.height || altitude <= 0) return null;
  const distance=Math.min((caster.height-target.height)/Math.tan(altitude),2000);
  const moved=turf.transformTranslate(caster.geometry,distance/1000,bearing,{units:'kilometers'});
  const points: Feature[]=[];
  turf.coordEach(caster.geometry,(coord) => points.push(turf.point(coord)));
  turf.coordEach(moved,(coord) => points.push(turf.point(coord)));
  return turf.convex(turf.featureCollection(points)) as PolygonFeature | null;
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
  const sunSamples = seasonalDates.map(([month,day]) => Array.from({length:24}, (_, hour) => {
    const sun=SunCalc.getPosition(new Date(Date.UTC(2026,month,day,hour-9,30)),lat,lon);
    if (sun.altitude<=0) return null;
    const sine=Math.sin(sun.altitude);
    return {altitude:sun.altitude, bearing:(sun.azimuth*180/Math.PI+360)%360, weight:1098*sine*Math.exp(-.057/sine)};
  }).filter((sample): sample is NonNullable<typeof sample> => sample !== null));
  const mapped: Building[]=data.features.map((ft,i) => ({geometry:ft as PolygonFeature,height:ft.properties.heightM,name:ft.properties.name,id:`building-${i}`,source:ft.properties.heightSource}));
  const bounds=mapped.map((item)=>turf.bbox(item.geometry));
  const nearby=mapped.map((a, i) => {
    const buffer=turf.buffer(a.geometry,0.3,{units:'kilometers'});
    if (!buffer) return [];
    const box=turf.bbox(buffer);
    return mapped.filter((b,j) => a!==b && bounds[j][0]<=box[2] && bounds[j][2]>=box[0] && bounds[j][1]<=box[3] && bounds[j][3]>=box[1] && turf.booleanIntersects(buffer,b.geometry));
  });
  const results: AnalysisResult={ buildings:empty(),existing_panels:empty(),available_roofs:empty(),summaries:[],warnings:[],totals:{availableAreaM2:null,panelAreaM2:null,capacityKw:null,annualEnergyKwh:null},models:{roof:null,panel:'NLR RTMDet-Ins'} };
  const panelGeometries=panels.map((panel) => { const clip=intersection(panel,roi); return clip ? {...clip,properties:panel.properties} : null; }).filter((p):p is NonNullable<typeof p> => p!==null);
  results.existing_panels.features = panelGeometries;
  if (!monthly) results.warnings.push('일사량 조회 실패: 발전량은 미계산입니다.');
  results.warnings.push('SAM 옥상 마스크와 건물 외곽선의 겹치는 부분을 평평한 옥상으로 가정하며 옥탑·설비·나무·경사는 반영하지 않습니다.');
  results.warnings.push('네 절기 낮 시간 그림자를 패널 줄 단위로 근사하고 월별 직달·산란 수평면 일사량과 성능계수 0.8을 사용합니다. 연간 일사량 70% 이상인 줄만 포함하며, 기준은 임의 가정입니다.');
  results.warnings.push('산란광 차폐, 패널 경사·전기적 부분 음영 손실은 반영하지 않아 실제 발전량과 차이가 날 수 있습니다.');
  const assumed=mapped.filter((b)=>b.source==='default').length;
  const fromFloors=mapped.filter((b)=>b.source==='floors').length;
  if (assumed) results.warnings.push(`높이가 없는 건물 ${assumed}개는 15m로 가정했습니다. 그림자와 발전량의 불확실성이 큽니다.`);
  if (fromFloors) results.warnings.push(`건물 ${fromFloors}개 높이는 지상 층수 × 3.3m로 추정했습니다.`);
  let missing=0, missingRoof=0;
  for (let index=0; index<mapped.length; index++) {
    const building=mapped[index], footprint=intersection(building.geometry,roi);
    if (!footprint) continue;
    const roofParts = roofs.filter((roof) => turf.booleanIntersects(footprint, roof));
    const joinedRoof = roofParts.length > 1 ? turf.union(turf.featureCollection(roofParts)) as PolygonFeature | null : roofParts[0] ?? null;
    const detectedRoof = joinedRoof ? intersection(footprint, joinedRoof) : null;
    if (!detectedRoof) {
      missingRoof++;
      const summary: Summary={buildingId:building.id,name:building.name,availableAreaM2:null,panelAreaM2:null,capacityKw:null,annualEnergyKwh:null,heightSource:building.source};
      results.summaries.push(summary);
      results.buildings.features.push({...footprint,properties:summary});
      continue;
    }
    let usable: PolygonFeature | undefined=detectedRoof;
    for (const panel of panelGeometries) if (usable && turf.booleanIntersects(usable,panel)) usable=(turf.difference(turf.featureCollection([usable,panel])) ?? undefined) as PolygonFeature | undefined;
    if (usable) usable=turf.buffer(usable,-0.001,{units:'kilometers'}) as PolygonFeature | undefined;
    const area=usable ? turf.area(usable) : 0;
    const cells=usable && area>0 ? modules(usable,detectedRoof) : [];
    const missingHeight=building.height===null || nearby[index].some((item)=>item.height===null);
    let capacity: number | null=null, energy: number | null=null, panelArea: number | null=null;
    if (missingHeight) missing++;
    if (usable) results.available_roofs.features.push({...usable,properties:{buildingId:building.id,areaM2:Math.round(area*100)/100}});
    const nearbyBuildings=nearby[index];
    if (!missingHeight && monthly && diffuse) {
      const weights=sunSamples.map((samples)=>samples.reduce((sum,sample)=>sum+sample.weight,0));
      const unshaded=Array.from({length:cells.length},()=>Array(4).fill(0) as number[]);
      const cellBounds=cells.map((cell)=>turf.bbox(cell.geometry));
      const cellAreas=cells.map((cell)=>turf.area(cell.geometry));
      const usableBounds=cells.length ? turf.bbox(turf.featureCollection(cells.map((cell)=>cell.geometry))) : null;
      for (let season=0;season<sunSamples.length;season++) for (const sample of sunSamples[season]) {
        const {altitude,bearing,weight}=sample;
        const shadows=nearbyBuildings.map((n)=>shadow(n,building,altitude,bearing)).filter((p):p is PolygonFeature=>p!==null && usableBounds!==null && boxesOverlap(turf.bbox(p),usableBounds));
        const merged = shadows.length > 1 ? turf.union(turf.featureCollection(shadows)) as PolygonFeature | null : shadows[0] ?? null;
        const shadowBounds=merged ? turf.bbox(merged) : null;
        for (let i=0;i<cells.length;i++) {
          const overlap=merged && shadowBounds && boxesOverlap(cellBounds[i],shadowBounds) ? intersection(cells[i].geometry,merged) : null;
          const shade=overlap ? turf.area(overlap) : 0;
          unshaded[i][season]+=Math.max(0,1-Math.min(1,shade/cellAreas[i]))*weight;
        }
      }
      const totalIrradiance=monthly.reduce((a,b)=>a+b,0);
      const yields=unshaded.map((samples)=>annualRowYield(monthly,diffuse,monthlyShadeFactors(samples.map((v,s)=>weights[s]?v/weights[s]:0))));
      const valid=cells.filter((_,i)=>yields[i]/totalIrradiance>=.7);
      const moduleCount=valid.reduce((sum,row)=>sum+row.count,0);
      panelArea=moduleCount*2;
      capacity=.4*moduleCount;
      energy=cells.reduce((sum,row,i)=>sum+(yields[i]/totalIrradiance>=.7 ? row.count*.4*.8*yields[i] : 0),0);
    }
    const summary: Summary={ buildingId:building.id,name:building.name,availableAreaM2:Math.round(area*100)/100,panelAreaM2:panelArea,capacityKw:capacity===null?null:Math.round(capacity*100)/100,annualEnergyKwh:energy===null?null:Math.round(energy*10)/10,heightSource:building.source };
    results.summaries.push(summary);
    results.buildings.features.push({...footprint,properties:summary});
    progress(`건물 면적·그림자 계산 중 (${results.summaries.length})`);
    await new Promise((resolve)=>setTimeout(resolve,0));
  }
  if (missing) results.warnings.push(`건물 ${missing}개는 주변 건물 높이 누락으로 발전량을 계산하지 못했습니다.`);
  if (missingRoof) results.warnings.push(`건물 ${missingRoof}개는 선택한 모델의 옥상 탐지 결과가 없어 계산하지 못했습니다.`);
  for (const key of ['availableAreaM2','panelAreaM2','capacityKw','annualEnergyKwh'] as const) {
    const values=results.summaries.map((item)=>item[key]);
    results.totals[key]=values.some((v)=>v===null)?null:Math.round(values.reduce<number>((sum,v)=>sum+(v ?? 0),0)*100)/100;
  }
  return results;
}
