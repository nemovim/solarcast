import test from 'node:test';
import assert from 'node:assert/strict';
import { loadAvailable, loadPrecomputed, loadRoofs, readModels, viewportTiles, coverageGeometry } from '../app/client-precomputed';
import { worldCoordinate } from '../app/client-imagery';
import type { Viewport } from '../app/map-panel';

const x=446793, y=207630, key=`${x}_${y}`;
const viewport: Viewport=[worldCoordinate(x*256+10,y*256+10),worldCoordinate(x*256+200,y*256+10),worldCoordinate(x*256+200,y*256+200),worldCoordinate(x*256+10,y*256+200)];
const manifest={version:1,zoom:19,model:'test',model_sha256:'test',min_confidence:.05,source:'test',imagery_observed_at:null,updated_at:'2026-01-01',tiles:{[key]:{file:`tiles/${key}.geojson`,count:2,processed_at:'2026-01-01'}}};
const polygon={type:'Polygon',coordinates:[[[126,35],[126.1,35],[126.1,35.1],[126,35]]]};

test('selects only processed tiles and filters precomputed detection confidence', async () => {
  const original=globalThis.fetch;
  globalThis.fetch=async (url)=>String(url).endsWith('manifest.json') ? Response.json(manifest) : Response.json({type:'FeatureCollection',features:[
    {type:'Feature',geometry:polygon,properties:{confidence:.3}},
    {type:'Feature',geometry:polygon,properties:{confidence:.8}},
  ]});
  try {
    assert.deepEqual(viewportTiles(viewport),[key]);
    assert.equal(coverageGeometry(manifest).features.length,1);
    const result=await loadPrecomputed('model',viewport,.5,()=>{});
    assert.equal(result.panels.length,1);
    assert.equal(result.panels[0].properties.confidence,.8);
    await assert.rejects(()=>loadPrecomputed('model',viewport,.01,()=>{}),/신뢰도/);
  } finally { globalThis.fetch=original; }
});

test('loads only tiles present in both selected models', async () => {
  const original=globalThis.fetch;
  const otherKey=`${x+1}_${y}`;
  const wide: Viewport=[worldCoordinate(x*256+10,y*256+10),worldCoordinate((x+1)*256+200,y*256+10),worldCoordinate((x+1)*256+200,y*256+200),worldCoordinate(x*256+10,y*256+200)];
  const tile = (tileKey: string) => ({file:`tiles/${tileKey}.geojson`,count:0,processed_at:'2026-01-01'});
  const urls: string[]=[];
  globalThis.fetch=async (url)=> {
    const path=String(url); urls.push(path);
    if (path.endsWith('manifest.json')) return Response.json({...manifest, tiles:path.includes('/panel/') ? {[key]:tile(key),[otherKey]:tile(otherKey)} : {[key]:tile(key)}});
    return Response.json({type:'FeatureCollection',features:[]});
  };
  try {
    const result = await loadAvailable('panel','roof',wide,.5,()=>{});
    assert.equal(result.processed,1);
    assert.equal(result.partial,true);
    assert.equal(result.coverage.features.length,1);
    assert.ok(!urls.some((url)=>url.endsWith(`${otherKey}.geojson`)));
    const distant: Viewport=[worldCoordinate((x-30)*256,(y-30)*256),worldCoordinate((x+30)*256,(y-30)*256),worldCoordinate((x+30)*256,(y+30)*256),worldCoordinate((x-30)*256,(y+30)*256)];
    const wideResult = await loadAvailable('panel','roof',distant,.5,()=>{});
    assert.equal(wideResult.processed,1);
    assert.equal(wideResult.partial,true);
  } finally { globalThis.fetch=original; }
});

test('reports when the two models have no common processed tile', async () => {
  const original=globalThis.fetch;
  globalThis.fetch=async (url)=>Response.json({...manifest,tiles:String(url).includes('/panel/') ? manifest.tiles : {}});
  try { await assert.rejects(()=>loadAvailable('panel','roof',viewport,.5,()=>{}),/겹치는 영역이 없습니다/); }
  finally { globalThis.fetch=original; }
});

test('missing data is not confused with a processed tile with no detections', async () => {
  const original=globalThis.fetch;
  globalThis.fetch=async (url)=>String(url).endsWith('manifest.json') ? Response.json({...manifest,tiles:{}}) : Response.json({type:'FeatureCollection',features:[]});
  try { await assert.rejects(()=>loadPrecomputed('model',viewport,.5,()=>{}),/사전 추론 결과가 없습니다/); }
  finally { globalThis.fetch=original; }
  globalThis.fetch=async (url)=>String(url).endsWith('manifest.json') ? Response.json({...manifest,tiles:{[key]:{...manifest.tiles[key],count:0}}}) : Response.json({type:'FeatureCollection',features:[]});
  try { assert.equal((await loadPrecomputed('model',viewport,.5,()=>{})).panels.length,0); }
  finally { globalThis.fetch=original; }
});

test('loads the selected model and its tiles from its own folder', async () => {
  const original=globalThis.fetch;
  const urls: string[]=[];
  globalThis.fetch=async (url)=> {
    const path=String(url); urls.push(path);
    if (path.endsWith('index.json')) return Response.json({version:1,models:[{id:'first',name:'first'},{id:'second',name:'second'}]});
    if (path.endsWith('manifest.json')) return Response.json({...manifest,model:'second',tiles:{[key]:{...manifest.tiles[key],count:0}}});
    return Response.json({type:'FeatureCollection',features:[]});
  };
  try {
    assert.equal((await readModels('panel')).length,2);
    assert.equal((await loadPrecomputed('second',viewport,.5,()=>{})).manifest.model,'second');
    assert.ok(urls.includes('/precomputed/panel/second/manifest.json'));
    assert.ok(urls.includes(`/precomputed/panel/second/tiles/${key}.geojson`));
    const roof = await loadRoofs('second',viewport,()=>{});
    assert.equal(roof.roofs.length, 0);
    assert.ok(urls.includes('/precomputed/roof/second/manifest.json'));
    assert.equal(coverageGeometry(manifest,{...manifest,tiles:{}}).features.length,0);
    await assert.rejects(()=>loadPrecomputed('../first',viewport,.5,()=>{}),/이름이 올바르지/);
  } finally { globalThis.fetch=original; }
});
