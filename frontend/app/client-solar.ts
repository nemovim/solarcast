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
const empty = (): FeatureCollection<Polygon | MultiPolygon> => ({ type:'FeatureCollection', features:[] });

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

function modules(candidate: PolygonFeature, roof: PolygonFeature): PolygonFeature[] {
  const center=turf.centroid(roof).geometry.coordinates;
  const angle=direction(roof,center[1]);
  const aligned=turf.transformRotate(candidate,-angle,{pivot:center}) as PolygonFeature;
  const [minX,minY,maxX,maxY]=turf.bbox(aligned);
  const dx=1/(111320*Math.cos(center[1]*Math.PI/180)), dy=1/110574;
  if ((maxX-minX)/dx*(maxY-minY)/dy > 2_000_000) throw new Error('건물 면적이 계산 가능한 범위를 넘었습니다.');
  let best: PolygonFeature[]=[];
  for (let py=0;py<4;py++) for (let px=0;px<4;px++) {
    const current: PolygonFeature[]=[];
    for (let y=minY+py*.75*dy; y+2*dy<=maxY; y+=3*dy) {
      let run: PolygonFeature[]=[];
      for (let x=minX+px*.25*dx; x<=maxX; x+=dx) {
        const cell=turf.bboxPolygon([x,y,x+dx,y+2*dy]) as PolygonFeature;
        if (x+dx<=maxX && turf.booleanContains(aligned,cell)) run.push(cell);
        else { if (run.length>=3) current.push(...run); run=[]; }
      }
      if (run.length>=3) current.push(...run);
    }
    if (current.length>best.length) best=current;
  }
  return best.map((cell) => turf.transformRotate(cell,angle,{pivot:center}) as PolygonFeature);
}

function shadow(caster: Building, target: Building, altitude: number, bearing: number): PolygonFeature | null {
  if (caster.height === null || target.height === null || caster.height <= target.height || altitude <= 3*Math.PI/180) return null;
  const distance=Math.min((caster.height-target.height)/Math.tan(altitude),2000);
  const moved=turf.transformTranslate(caster.geometry,distance/1000,bearing,{units:'kilometers'});
  const points: Feature[]=[];
  turf.coordEach(caster.geometry,(coord) => points.push(turf.point(coord)));
  turf.coordEach(moved,(coord) => points.push(turf.point(coord)));
  return turf.convex(turf.featureCollection(points)) as PolygonFeature | null;
}

export async function estimateSolar(viewport: Viewport, panels: PanelFeature[], roofs: RoofFeature[], shadowAt: string, progress: (s:string)=>void, coverage?: FeatureCollection<Polygon>): Promise<AnalysisResult> {
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
  const monthly = irradianceResponse.ok ? (await irradianceResponse.json()).monthly as number[] : null;
  const mapped: Building[]=data.features.map((ft,i) => ({geometry:ft as PolygonFeature,height:ft.properties.heightM,name:ft.properties.name,id:`building-${i}`,source:ft.properties.heightSource}));
  const bounds=mapped.map((item)=>turf.bbox(item.geometry));
  const nearby=mapped.map((a, i) => {
    const buffer=turf.buffer(a.geometry,0.3,{units:'kilometers'});
    if (!buffer) return [];
    const box=turf.bbox(buffer);
    return mapped.filter((b,j) => a!==b && bounds[j][0]<=box[2] && bounds[j][2]>=box[0] && bounds[j][1]<=box[3] && bounds[j][3]>=box[1] && turf.booleanIntersects(buffer,b.geometry));
  });
  const panelGeometries=panels.map((ft)=>intersection(ft,roi)).filter((ft):ft is PolygonFeature => ft!==null);
  const results: AnalysisResult={ buildings:empty(),existing_panels:empty(),available_roofs:empty(),shadows:empty(),summaries:[],warnings:[],totals:{availableAreaM2:null,capacityKw:null,annualEnergyKwh:null},models:{roof:null,panel:'NLR RTMDet-Ins'} };
  results.existing_panels.features = panels.map((panel) => { const clip=intersection(panel,roi); return clip ? {...clip,properties:panel.properties} : null; }).filter((p):p is NonNullable<typeof p> => p!==null);
  if (!monthly) results.warnings.push('일사량 조회 실패: 발전량은 미계산입니다.');
  results.warnings.push('SAM 옥상 마스크와 건물 외곽선의 겹치는 부분을 평평한 옥상으로 가정하며 옥탑·설비·나무·경사는 반영하지 않습니다.');
  results.warnings.push('인접 건물의 높이를 바탕으로 그림자를 단순 추정하고, 월별 시간대 표본과 수평면 일사량으로 발전량을 계산합니다.');
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
      const summary: Summary={buildingId:building.id,name:building.name,availableAreaM2:null,capacityKw:null,annualEnergyKwh:null,heightSource:building.source};
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
    let capacity: number | null=null, energy: number | null=null;
    if (missingHeight) missing++;
    if (usable) results.available_roofs.features.push({...usable,properties:{buildingId:building.id,areaM2:Math.round(area*100)/100}});
    const nearbyBuildings=nearby[index];
    if (!missingHeight && monthly) {
      const weights=Array.from({length:12},()=>0);
      const unshaded=Array.from({length:cells.length},()=>Array(12).fill(0) as number[]);
      for (let month=0;month<12;month++) for (let hour=5;hour<20;hour++) {
        const time=new Date(Date.UTC(2025,month,21,hour-9,30));
        const sun=SunCalc.getPosition(time,lat,lon);
        if (sun.altitude<10*Math.PI/180) continue;
        const sine=Math.sin(sun.altitude), weight=1098*sine*Math.exp(-.057/sine);
        weights[month]+=weight;
        const bearing=(sun.azimuth*180/Math.PI+360)%360;
        const shadows=nearbyBuildings.map((n)=>shadow(n,building,sun.altitude,bearing)).filter((p):p is PolygonFeature=>p!==null);
        const merged = shadows.length > 1 ? turf.union(turf.featureCollection(shadows)) as PolygonFeature | null : shadows[0] ?? null;
        for (let i=0;i<cells.length;i++) {
          const overlap=merged ? intersection(cells[i],merged) : null;
          const shade=overlap ? turf.area(overlap) : 0;
          unshaded[i][month]+=Math.max(0,1-Math.min(1,shade/turf.area(cells[i])))*weight;
        }
      }
      const valid=unshaded.map((samples)=>samples.map((v,m)=>weights[m]?v/weights[m]:0)).filter((sample)=>sample.reduce((sum,v,m)=>sum+v*monthly[m],0)/monthly.reduce((a,b)=>a+b,0)>=.7);
      capacity=.4*valid.length;
      energy=valid.reduce((sum,sample)=>sum+.4*.8*sample.reduce((acc,v,m)=>acc+v*monthly[m],0),0);
    }
    if (!missingHeight && usable) {
      const sun=SunCalc.getPosition(new Date(shadowAt),lat,lon), bearing=(sun.azimuth*180/Math.PI+360)%360;
      for (const other of nearbyBuildings) {
        const sh=shadow(other,building,sun.altitude,bearing);
        const overlap=sh ? intersection(sh,usable) : null;
        if (overlap) results.shadows.features.push({...overlap,properties:{buildingId:building.id,time:shadowAt}});
      }
    }
    const summary: Summary={ buildingId:building.id,name:building.name,availableAreaM2:Math.round(area*100)/100,capacityKw:capacity===null?null:Math.round(capacity*100)/100,annualEnergyKwh:energy===null?null:Math.round(energy*10)/10,heightSource:building.source };
    results.summaries.push(summary);
    results.buildings.features.push({...footprint,properties:summary});
    progress(`건물 면적·그림자 계산 중 (${results.summaries.length})`);
    await new Promise((resolve)=>setTimeout(resolve,0));
  }
  if (missing) results.warnings.push(`건물 ${missing}개는 주변 건물 높이 누락으로 발전량을 계산하지 못했습니다.`);
  if (missingRoof) results.warnings.push(`건물 ${missingRoof}개는 선택한 모델의 옥상 탐지 결과가 없어 계산하지 못했습니다.`);
  for (const key of ['availableAreaM2','capacityKw','annualEnergyKwh'] as const) {
    const values=results.summaries.map((item)=>item[key]);
    results.totals[key]=values.some((v)=>v===null)?null:Math.round(values.reduce<number>((sum,v)=>sum+(v ?? 0),0)*100)/100;
  }
  return results;
}
