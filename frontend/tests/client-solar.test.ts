import test from 'node:test';
import assert from 'node:assert/strict';
import { annualRowYield, estimateSolar, monthlyShadeFactors } from '../app/client-solar';
import type { Viewport } from '../app/map-panel';
import { worldCoordinate, worldPixel, extent } from '../app/client-imagery';

test('annual irradiance preserves diffuse light while shading direct light', () => {
  const ghi=Array(12).fill(100), diffuse=Array(12).fill(40);
  assert.equal(annualRowYield(ghi,diffuse,Array(12).fill(1)),1200);
  assert.equal(annualRowYield(ghi,diffuse,Array(12).fill(0)),480);
  assert.equal(annualRowYield(ghi,diffuse,Array(12).fill(.5)),840);
});

test('monthly shade factors interpolate between seasonal days across the year boundary', () => {
  const values=monthlyShadeFactors([0,1,0,1]);
  assert.equal(values.length,12);
  assert.ok(values.every((value)=>value>=0 && value<=1));
  assert.ok(values[0]>values[1] && values[1]>values[2]);
  assert.ok(values[3]<values[4] && values[4]<values[5]);
  assert.ok(monthlyShadeFactors([0.4,0.4,0.4,0.4]).every((value)=>Math.abs(value-0.4)<1e-12));
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
    assert.equal(target.panelAreaM2! * .2,target.capacityKw);
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
