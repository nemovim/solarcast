import test from 'node:test';
import assert from 'node:assert/strict';
import { annualYield, buildSky, estimateSolar, horizonProfile } from '../app/client-solar';
import type { Viewport } from '../app/map-panel';
import { worldCoordinate, worldPixel, extent } from '../app/client-imagery';

test('sky model splits monthly irradiation without losing energy', () => {
  const monthly=Array(12).fill(100), diffuse=Array(12).fill(40);
  const sky=buildSky(35.0218,126.7885,monthly,diffuse);
  const binned=sky.bins.reduce((sum,bin)=>sum+(bin.above[0] ?? 0),0);
  assert.ok(Math.abs(binned-720)<1e-9);
  assert.equal(sky.global,1200);
  assert.ok(Math.abs(annualYield(sky,new Float64Array(120))-1200)<1e-9);
  // Southern bins carry far more direct light than northern ones at this latitude.
  assert.ok(sky.bins[60].above[0] > 10*(sky.bins[0].above[0] ?? 0));
});

test('horizon lookup applies the visible sun and isotropic sky view factor', () => {
  const sky=buildSky(35.0218,126.7885,Array(12).fill(100),Array(12).fill(40));
  const h=Math.tan(Math.PI/6);
  const expectedDirect=sky.bins.reduce((sum,bin)=>sum+bin.tan.reduce((acc,t,i)=>acc+(t>h ? (bin.above[i]-(bin.above[i+1] ?? 0)) : 0),0),0);
  const value=annualYield(sky,new Float64Array(120).fill(h));
  assert.ok(Math.abs(value-(expectedDirect+480*.75))<1e-9);
  assert.ok(Math.abs(annualYield(sky,new Float64Array(120).fill(1e9)))<1e-6);
});

test('horizon profile finds the steepest edge in each azimuth bin', () => {
  // A 10 m tall wall 10 m south of the point, 40 m wide; a lower block 20 m east.
  const wall={rings:[[[-20,-10],[20,-10],[20,-14],[-20,-14],[-20,-10]]],rise:10};
  const block={rings:[[[20,-2],[24,-2],[24,2],[20,2],[20,-2]]],rise:5};
  const horizon=horizonProfile(0,0,[wall,block]);
  assert.ok(Math.abs(horizon[60]-1)<1e-9);
  assert.ok(Math.abs(horizon[30]-.25)<1e-9);
  assert.equal(horizon[0],0);
  assert.equal(horizon[90],0);
  const sky=buildSky(35.0218,126.7885,Array(12).fill(100),Array(12).fill(40));
  const shaded=annualYield(sky,horizon);
  assert.ok(shaded<1200 && shaded>480);
});

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
    ? Response.json({monthly:Array(12).fill(100),diffuse:Array(12).fill(40)}) : Response.json(buildings);
  const viewport: Viewport=[[lon-2*dx,lat+14*dy],[lon+14*dx,lat+14*dy],[lon+14*dx,lat-12*dy],[lon-2*dx,lat-12*dy]];
  const panel={type:'Feature',geometry:{type:'MultiPolygon',coordinates:[box(4,4,7,7).coordinates]},properties:{confidence:.9}} as const;
  try {
    const roofs = buildings.features as never;
    const bare=await estimateSolar(viewport,[],roofs,()=>{});
    const result=await estimateSolar(viewport,[panel as never],roofs,()=>{});
    const target=result.summaries.find((b)=>b.name==='대상')!;
    const baseline=bare.summaries.find((b)=>b.name==='대상')!;
    globalThis.fetch=async (input) => String(input).includes('/api/irradiance')
      ? Response.json({monthly:Array(12).fill(100),diffuse:Array(12).fill(40)}) : Response.json({...buildings,features:[buildings.features[0]]});
    const unshaded=await estimateSolar(viewport,[],roofs,()=>{});
    assert.ok(baseline.annualEnergyKwh! < unshaded.summaries[0].annualEnergyKwh!);
    globalThis.fetch=async (input) => String(input).includes('/api/irradiance')
      ? Response.json({monthly:Array(12).fill(100),diffuse:Array(12).fill(40)}) : Response.json(buildings);
    assert.ok(target.availableAreaM2! < baseline.availableAreaM2!);
    assert.notEqual(target.annualEnergyKwh,null);
    assert.ok(Math.abs(target.panelAreaM2! * .2-target.capacityKw!)<1e-9);
    assert.ok(target.panelAreaM2! <= target.availableAreaM2!);
    const smallerRoof = [{type:'Feature',geometry:box(0,0,8,8),properties:{}}];
    const smaller = await estimateSolar(viewport,[],smallerRoof as never,()=>{});
    assert.ok(smaller.summaries.find((b)=>b.name==='대상')!.availableAreaM2! < baseline.availableAreaM2!);
    const coverage = {type:'FeatureCollection',features:[{type:'Feature',geometry:box(0,-1,6,14),properties:{}}]} as never;
    const partial = await estimateSolar(viewport,[],roofs,()=>{},coverage);
    assert.ok(partial.summaries.find((b)=>b.name==='대상')!.availableAreaM2! < baseline.availableAreaM2!);
    assert.ok(!partial.summaries.some((b)=>b.name==='인접'));
    const noRoof=await estimateSolar(viewport,[],[],()=>{});
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
  globalThis.fetch=async (input)=>String(input).includes('/api/irradiance') ? Response.json({monthly:Array(12).fill(100),diffuse:Array(12).fill(40)}) : Response.json(buildings);
  try {
    const result=await estimateSolar(viewport,[],buildings.features as never,()=>{});
    assert.ok(result.summaries[0].capacityKw! > 0);
    assert.ok(result.summaries[0].annualEnergyKwh! > 0);
    assert.ok(result.warnings.some((message)=>message.includes('15m')));
    assert.ok(Math.abs(result.summaries[0].annualEnergyKwh! - result.summaries[0].capacityKw! * .8 * 1200) < .11);
  } finally { globalThis.fetch=realFetch; }
});

test('missing diffuse data leaves generation and shade-qualified panel area uncalculated', async () => {
  const lon=126.7885, lat=35.0218;
  const viewport: Viewport=[[lon-.0003,lat+.0003],[lon+.0003,lat+.0003],[lon+.0003,lat-.0003],[lon-.0003,lat-.0003]];
  const buildings={type:'FeatureCollection',features:[{type:'Feature',geometry:{type:'Polygon',coordinates:[[[lon-.0001,lat-.0001],[lon+.0001,lat-.0001],[lon+.0001,lat+.0001],[lon-.0001,lat+.0001],[lon-.0001,lat-.0001]]]},properties:{name:'건물',heightM:15,heightSource:'record'}}]};
  const original=globalThis.fetch;
  globalThis.fetch=async (input)=>String(input).includes('/api/irradiance') ? Response.json({monthly:Array(12).fill(100)}) : Response.json(buildings);
  try {
    const result=await estimateSolar(viewport,[],buildings.features as never,()=>{});
    assert.ok(result.summaries[0].availableAreaM2!>0);
    assert.equal(result.summaries[0].panelAreaM2,null);
    assert.equal(result.summaries[0].annualEnergyKwh,null);
  } finally { globalThis.fetch=original; }
});

test('totals add computed buildings and count the ones left out', async () => {
  const lon=126.7885, lat=35.0218;
  const dx=1/(111320*Math.cos(lat*Math.PI/180)), dy=1/110574;
  const box=(x0:number,y0:number,x1:number,y1:number) => ({ type:'Polygon', coordinates:[[
    [lon+x0*dx,lat+y0*dy],[lon+x1*dx,lat+y0*dy],[lon+x1*dx,lat+y1*dy],[lon+x0*dx,lat+y1*dy],[lon+x0*dx,lat+y0*dy],
  ]] });
  const buildings={type:'FeatureCollection',features:[
    {type:'Feature',geometry:box(0,0,12,12),properties:{name:'옥상 있음',heightM:15,heightSource:'record'}},
    {type:'Feature',geometry:box(30,0,42,12),properties:{name:'옥상 없음',heightM:15,heightSource:'record'}},
  ]};
  const realFetch=globalThis.fetch;
  globalThis.fetch=async (input) => String(input).includes('/api/irradiance')
    ? Response.json({monthly:Array(12).fill(100),diffuse:Array(12).fill(40)}) : Response.json(buildings);
  const viewport: Viewport=[[lon-2*dx,lat+14*dy],[lon+44*dx,lat+14*dy],[lon+44*dx,lat-2*dy],[lon-2*dx,lat-2*dy]];
  try {
    const result=await estimateSolar(viewport,[],[buildings.features[0]] as never,()=>{});
    const computed=result.summaries.find((b)=>b.name==='옥상 있음')!;
    assert.equal(result.totals.availableAreaM2,computed.availableAreaM2);
    assert.equal(result.totals.annualEnergyKwh,computed.annualEnergyKwh);
    assert.deepEqual(result.excluded,{availableAreaM2:1,panelAreaM2:1,capacityKw:1,annualEnergyKwh:1});
    const none=await estimateSolar(viewport,[],[],()=>{});
    assert.equal(none.totals.availableAreaM2,null);
  } finally { globalThis.fetch=realFetch; }
});

test('virtual modules are not placed over holes in the usable roof', async () => {
  const lon=126.7885, lat=35.0218;
  const dx=1/(111320*Math.cos(lat*Math.PI/180)), dy=1/110574;
  const ring=(x0:number,y0:number,x1:number,y1:number) => [[lon+x0*dx,lat+y0*dy],[lon+x1*dx,lat+y0*dy],[lon+x1*dx,lat+y1*dy],[lon+x0*dx,lat+y1*dy],[lon+x0*dx,lat+y0*dy]];
  const buildings={type:'FeatureCollection',features:[{type:'Feature',geometry:{type:'Polygon',coordinates:[ring(0,0,20,20)]},properties:{name:'건물',heightM:15,heightSource:'record'}}]};
  // A 0.5 m hole (two grid cells) grows to about 2.5 m with the 1 m setback; modules that would
  // only partly cover it must still be rejected.
  const solid={type:'Feature',geometry:{type:'Polygon',coordinates:[ring(0,0,20,20)]},properties:{}};
  const holed={type:'Feature',geometry:{type:'Polygon',coordinates:[ring(0,0,20,20),ring(9.75,9.75,10.25,10.25).reverse()]},properties:{}};
  const realFetch=globalThis.fetch;
  globalThis.fetch=async (input) => String(input).includes('/api/irradiance')
    ? Response.json({monthly:Array(12).fill(100),diffuse:Array(12).fill(40)}) : Response.json(buildings);
  const viewport: Viewport=[[lon-2*dx,lat+22*dy],[lon+22*dx,lat+22*dy],[lon+22*dx,lat-2*dy],[lon-2*dx,lat-2*dy]];
  try {
    const full=(await estimateSolar(viewport,[],[solid] as never,()=>{})).summaries[0];
    const withHole=(await estimateSolar(viewport,[],[holed] as never,()=>{})).summaries[0];
    assert.ok(withHole.panelAreaM2! < full.panelAreaM2!);
    assert.ok(withHole.panelAreaM2! <= withHole.availableAreaM2!);
  } finally { globalThis.fetch=realFetch; }
});
