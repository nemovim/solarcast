import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source = await readFile(new URL('../web/geometry.js', import.meta.url), 'utf8');
const {polygonError, translatePolygon, rectangleFromDiagonal, shuffledItems} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('random order preserves all assignments and repeats after reload',()=>{
  const items=Array.from({length:853},(_,id)=>({id,part:2}));
  const order=shuffledItems(items,'dataset:part-2');
  assert.equal(new Set(order.map(item=>item.id)).size,853);
  assert.deepEqual([...order].sort((a,b)=>a.id-b.id),items);
  assert.deepEqual(order,shuffledItems(items,'dataset:part-2'));
  assert.notDeepEqual(order,items);
  assert.notDeepEqual(order,shuffledItems(items,'other-dataset:part-2'));
  assert.equal(items[0].id,0);
  assert.deepEqual(shuffledItems([],'empty'),[]);
});

test('dragging in any diagonal direction creates the same ordered rectangle',()=>{
  for(const [a,b] of [[[10,20],[70,50]],[[70,50],[10,20]],[[10,50],[70,20]],[[70,20],[10,50]]]) {
    const points=rectangleFromDiagonal(a,b);
    assert.deepEqual(points,[[10,20],[70,20],[70,50],[10,50]]);
    assert.equal(polygonError(points),'');
  }
});
test('clicks and horizontal/vertical lines are not valid rectangles',()=>{
  for(const end of [[10,20],[50,20],[10,60]]) assert.ok(polygonError(rectangleFromDiagonal([10,20],end)));
});

test('rotated quadrilateral and clipped panel polygon are valid',()=>{
  assert.equal(polygonError([[10,5],[50,10],[45,30],[5,25]]),'');
  assert.equal(polygonError([[0,0],[20,0],[30,10],[20,30],[0,20]]),'');
});
test('crossing edges, duplicate vertices and zero area are rejected',()=>{
  for(const p of [[[0,0],[5,5],[0,5],[5,0]],[[0,0],[1,1],[0,0]],[[0,0],[1,0],[2,0]]]) assert.ok(polygonError(p));
});
test('moving a region preserves shape and clamps to image edges',()=>{
  const points=[[10,10],[20,10],[20,20],[10,20]];
  assert.deepEqual(translatePolygon(points,100,-100,50,40),[[40,0],[50,0],[50,10],[40,10]]);
  assert.deepEqual(points,[[10,10],[20,10],[20,20],[10,20]]);
});
