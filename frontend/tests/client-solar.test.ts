import test from 'node:test';
import assert from 'node:assert/strict';
import { estimateSolar } from '../app/client-solar';
import type { Viewport } from '../app/map-panel';
import { worldCoordinate, worldPixel, extent } from '../app/client-imagery';

test('map coordinates round trip and reject out-of-region bounds', () => {
  const [x,y]=worldPixel(126.7885,35.0218);
  const [lon,lat]=worldCoordinate(x,y);
  assert.ok(Math.abs(lon-126.7885)<1e-9);
  assert.ok(Math.abs(lat-35.0218)<1e-9);
  assert.throws(()=>extent([[0,0],[1,0],[1,1],[0,1]]));
});

test('footprint estimate includes panel exclusion and neighboring shadows', async () => {
  const lon=126.7885, lat=35.0218;
  const dx=1/(111320*Math.cos(lat*Math.PI/180)), dy=1/110574;
  const box=(x0:number,y0:number,x1:number,y1:number) => ({ type:'Polygon', coordinates:[[
    [lon+x0*dx,lat+y0*dy],[lon+x1*dx,lat+y0*dy],[lon+x1*dx,lat+y1*dy],[lon+x0*dx,lat+y1*dy],[lon+x0*dx,lat+y0*dy],
  ]] });
  const buildings={type:'FeatureCollection',features:[
    {type:'Feature',geometry:box(0,0,12,12),properties:{name:'대상',heightM:15,heightSource:'record'}},
    {type:'Feature',geometry:box(0,-10,12,-1),properties:{name:'인접',heightM:40,heightSource:'record'}},
  ]};
  const realFetch=globalThis.fetch;
  globalThis.fetch=async (input) => String(input).includes('/api/irradiance')
    ? Response.json({monthly:Array(12).fill(100)}) : Response.json(buildings);
  const viewport: Viewport=[[lon-2*dx,lat+14*dy],[lon+14*dx,lat+14*dy],[lon+14*dx,lat-12*dy],[lon-2*dx,lat-12*dy]];
  const panel={type:'Feature',geometry:{type:'MultiPolygon',coordinates:[box(4,4,7,7).coordinates]},properties:{confidence:.9}} as const;
  try {
    const roofs = buildings.features as never;
    const bare=await estimateSolar(viewport,[],roofs,'2026-06-21T12:00:00+09:00',()=>{});
    const result=await estimateSolar(viewport,[panel as never],roofs,'2026-06-21T12:00:00+09:00',()=>{});
    const target=result.summaries.find((b)=>b.name==='대상')!;
    const baseline=bare.summaries.find((b)=>b.name==='대상')!;
    assert.ok(target.availableAreaM2! < baseline.availableAreaM2!);
    assert.notEqual(target.annualEnergyKwh,null);
    assert.ok(result.shadows.features.some((f)=>f.properties?.buildingId===target.buildingId));
    const smallerRoof = [{type:'Feature',geometry:box(0,0,8,8),properties:{}}];
    const smaller = await estimateSolar(viewport,[],smallerRoof as never,'2026-06-21T12:00:00+09:00',()=>{});
    assert.ok(smaller.summaries.find((b)=>b.name==='대상')!.availableAreaM2! < baseline.availableAreaM2!);
    const coverage = {type:'FeatureCollection',features:[{type:'Feature',geometry:box(0,-1,6,14),properties:{}}]} as never;
    const partial = await estimateSolar(viewport,[],roofs,'2026-06-21T12:00:00+09:00',()=>{},coverage);
    assert.ok(partial.summaries.find((b)=>b.name==='대상')!.availableAreaM2! < baseline.availableAreaM2!);
    assert.ok(!partial.summaries.some((b)=>b.name==='인접'));
    const noRoof=await estimateSolar(viewport,[],[],'2026-06-21T12:00:00+09:00',()=>{});
    assert.equal(noRoof.summaries[0].availableAreaM2, null);
  } finally { globalThis.fetch=realFetch; }
});

test('missing building height uses disclosed 15m fallback', async () => {
  const lon=126.7885, lat=35.0218;
  const viewport: Viewport=[[lon-.0003,lat+.0003],[lon+.0003,lat+.0003],[lon+.0003,lat-.0003],[lon-.0003,lat-.0003]];
  const buildings={type:'FeatureCollection',features:[{type:'Feature',geometry:{type:'Polygon',coordinates:[[
    [lon-.0001,lat-.0001],[lon+.0001,lat-.0001],[lon+.0001,lat+.0001],[lon-.0001,lat+.0001],[lon-.0001,lat-.0001],
  ]]},properties:{name:'높이 미상',heightM:15,heightSource:'default'}}]};
  const realFetch=globalThis.fetch;
  globalThis.fetch=async (input)=>String(input).includes('/api/irradiance') ? Response.json({monthly:Array(12).fill(100)}) : Response.json(buildings);
  try {
    const result=await estimateSolar(viewport,[],buildings.features as never,'2026-06-21T12:00:00+09:00',()=>{});
    assert.ok(result.summaries[0].capacityKw! > 0);
    assert.ok(result.summaries[0].annualEnergyKwh! > 0);
    assert.ok(result.warnings.some((message)=>message.includes('15m')));
  } finally { globalThis.fetch=realFetch; }
});
