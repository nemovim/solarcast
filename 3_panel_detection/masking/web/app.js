import {clamp, polygonError, translatePolygon, rectangleFromDiagonal, shuffledItems} from './geometry.js';

const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const labels = {pending:'미작업',in_progress:'작업 중',done:'완료',empty:'패널 없음',deferred:'보류'};
const complete = status => status==='done'||status==='empty';
const copy = value => structuredClone(value);
let predictionRequest=0;
const state = {part:null,items:[],index:-1,record:null,tool:'quad',draft:[],selected:-1,
  undo:[],redo:[],dirty:false,version:0,saving:null,timer:null,busy:false,conflict:false,
  scale:1,offset:[0,0],drag:null,space:false,config:null,imageReady:false};

function notice(message='') { $('notice').textContent=message; $('notice').hidden=!message; }
function saveLabel(message, error=false) { $('saveState').textContent=message; $('saveState').classList.toggle('error',error); }
function storageGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
function storageSet(key,value) { try { localStorage.setItem(key,value); } catch { /* Server save still works. */ } }
function draftKey() { return `bitgaram-v2:${state.config.assignment_sha256}:${state.record.image_id}`; }
function backup() { if(state.record) storageSet(draftKey(),JSON.stringify(state.record)); }
function clearBackup() { try { localStorage.removeItem(draftKey()); } catch {} }
function hidePrediction() {
  predictionRequest++;$('aiOverlay').setAttribute('visibility','hidden');
  $('aiMaskImage').removeAttribute('href');$('aiStatus').textContent='';
}
async function showPrediction() {
  hidePrediction();
  if(!$('aiToggle').checked||!state.record)return;
  const token=predictionRequest,key=state.record.image_id;
  $('aiStatus').textContent='AI 예측 불러오는 중…';
  try {
    const result=await api(`/api/predictions/${key}`);
    if(token!==predictionRequest)return;
    if(!result.available){$('aiStatus').textContent=result.reason;return;}
    const img=new Image();img.src=result.url;await img.decode();
    if(token!==predictionRequest)return;
    if(img.naturalWidth!==state.record.width||img.naturalHeight!==state.record.height)throw new Error('AI 마스크 크기가 원본과 다릅니다.');
    for(const id of ['aiMask','aiMaskImage','aiOverlay']) {
      $(id).setAttribute('width',state.record.width);$(id).setAttribute('height',state.record.height);
    }
    $('aiMaskImage').setAttribute('href',img.src);$('aiOverlay').setAttribute('visibility','visible');
    $('aiStatus').textContent=`AI 예측 ${result.instances}개 · 주황색 · 참고용 (conf ≥ 0.1)`;
  } catch(error) {if(token===predictionRequest)$('aiStatus').textContent=`AI 예측 표시 실패: ${error.message}`;}
}
async function api(path, options) {
  const response=await fetch(path,options);
  const data=await response.json();
  if(!response.ok) { const error=new Error(data.error||'요청 실패'); error.status=response.status; throw error; }
  return data;
}
function snapshot() { return {polygons:copy(state.record.polygons),status:state.record.status}; }
function remember() { state.undo.push(snapshot()); if(state.undo.length>100) state.undo.shift(); state.redo=[]; }
function changed() {
  state.dirty=true;state.version++;backup();
  saveLabel('저장 대기…');clearTimeout(state.timer);
  state.timer=setTimeout(()=>flush().catch(()=>{}),400);
  renderStatus();renderList();updateControls();
}
function geometryChanged() { state.record.status='in_progress';changed();renderShapes();renderRegions(); }
async function flush() {
  clearTimeout(state.timer);
  if(state.saving) await state.saving;
  if(!state.dirty||!state.record) return;
  if(state.conflict) throw new Error('다른 창의 변경과 충돌했습니다. 초안을 내려받고 새로고침하세요.');
  state.saving=(async()=>{
    while(state.dirty) {
      const version=state.version, payload=copy(state.record);
      saveLabel('저장 중…');
      try {
        const saved=await api(`/api/records/${payload.image_id}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
        state.record.revision=saved.revision;state.record.updated_at=saved.updated_at;
        if(version===state.version) { state.dirty=false;clearBackup();saveLabel('로컬 파일에 저장됨'); }
        else backup();
        $('retry').hidden=true;$('downloadDraft').hidden=true;$('reloadRecord').hidden=true;
      } catch(error) {
        state.conflict=error.status===409;saveLabel('저장되지 않음',true);notice(error.message);
        $('retry').hidden=state.conflict;$('downloadDraft').hidden=false;$('reloadRecord').hidden=!state.conflict;updateControls();throw error;
      }
    }
  })();
  try { await state.saving; } finally { state.saving=null; }
}
async function run(action) {
  if(state.busy) return;
  state.busy=true;updateControls();
  try { await action(); } catch(error) { notice(error.message); }
  finally { state.busy=false;updateControls(); }
}
function requireNoDraft() {
  if(state.drag) { notice('드래그를 마치거나 Esc로 취소해 주세요.'); return false; }
  if(state.draft.length) { notice('그리던 영역을 완성하거나 Esc로 취소한 뒤 이동·저장해 주세요.'); return false; }
  return true;
}
async function selectPart(part) {
  if(!requireNoDraft()) return;
  await flush();
  const items=shuffledItems(await api(`/api/parts/${part}`),`${state.config.assignment_sha256}:part-${part}`);
  state.part=part;state.items=items;
  storageSet('bitgaram-v2:part',String(part));
  document.querySelectorAll('[data-part]').forEach(b=>b.classList.toggle('active',Number(b.dataset.part)===part));
  $('filter').value='all';$('search').value='';
  const last=storageGet(`bitgaram-v2:last:${part}`);
  let index=items.findIndex(i=>i.id===last);
  if(index<0) index=items.findIndex(i=>!complete(i.status));
  await loadImage(Math.max(0,index));
}
async function loadImage(index) {
  if(index<0||index>=state.items.length||!requireNoDraft()) return;
  await flush();
  const item=state.items[index];
  const record=await api(`/api/records/${item.id}`);
  // Do not replace the current editor until both its record and image are ready.
  const img=new Image();img.src=`/images/${item.id}`;
  await img.decode();
  if(img.naturalWidth!==record.width||img.naturalHeight!==record.height) throw new Error('원본 이미지 크기가 배정 목록과 다릅니다.');
  state.index=index;state.record=record;state.dirty=false;state.conflict=false;state.selected=-1;
  hidePrediction();
  state.undo=[];state.redo=[];state.draft=[];state.imageReady=true;notice();
  state.recovery=null;$('retry').hidden=true;$('downloadDraft').hidden=true;$('reloadRecord').hidden=true;
  const recovery=storageGet(draftKey());
  if(recovery) {
    try {
      const draft=JSON.parse(recovery);
      if(draft.revision===record.revision&&draft.image_id===record.image_id) {
        state.record=draft;state.dirty=true;state.version++;
        notice('지난번 저장되지 않은 로컬 초안을 복원했습니다. 다시 저장합니다.');
      } else {
        // Keep the recovery payload available without overwriting a newer server record.
        notice('이 이미지의 복구 초안과 저장된 기록의 버전이 다릅니다. 초안을 내려받아 확인한 뒤 편집하세요.');
        state.recovery=draft;state.conflict=true;$('downloadDraft').hidden=false;$('reloadRecord').hidden=false;
      }
    } catch { notice('로컬 복구 초안을 읽지 못했습니다. 서버에 저장된 기록을 불러왔습니다.'); }
  } else state.recovery=null;
  $('tile').setAttribute('href',img.src);$('tile').setAttribute('width',record.width);$('tile').setAttribute('height',record.height);
  $('welcome').hidden=true;$('imageTitle').textContent=item.id;
  $('note').value=state.record.note;$('position').textContent=`${index+1} / ${state.items.length}`;
  storageSet(`bitgaram-v2:last:${state.part}`,item.id);
  fit();renderStatus();renderList();renderRegions();
  showPrediction();
  saveLabel(state.record.revision?'로컬 파일에 저장됨':'아직 작업하지 않은 이미지');
  if(state.dirty) await flush();
}
function renderStatus() {
  if(state.record) {
    state.items[state.index].status=state.record.status;
    $('statusBadge').textContent=labels[state.record.status];
  }
  const done=state.items.filter(i=>complete(i.status)).length,total=state.items.length;
  const deferred=state.items.filter(i=>i.status==='deferred').length;
  $('progressText').textContent=total?`완료 ${done} / ${total} · ${Math.round(done/total*1000)/10}%`:'파트를 선택해 주세요';
  $('deferredText').textContent=`보류 ${deferred}`;
  $('progress').max=total||1;$('progress').value=done;
  $('imageSlider').max=total||1;$('imageSlider').value=state.index+1||1;
  $('imageSlider').setAttribute('aria-valuetext',total?`${state.index+1} / ${total}번째 이미지`:'파트를 선택하세요');
}
function renderList() {
  const fragment=document.createDocumentFragment(),filter=$('filter').value,query=$('search').value.trim();
  state.items.forEach((item,index)=>{
    if(!item.id.includes(query)||(filter==='unfinished'?complete(item.status):filter!=='all'&&item.status!==filter)) return;
    const button=document.createElement('button');button.className='image-row';button.dataset.status=item.status;button.dataset.index=index;
    button.classList.toggle('current',index===state.index);
    button.setAttribute('aria-label',`${item.id} ${labels[item.status]}`);
    const text=document.createElement('span');text.textContent=item.id;
    const badge=document.createElement('small');badge.textContent=labels[item.status];
    button.append(text,badge);fragment.append(button);
  });
  $('imageList').replaceChildren(fragment);
}
function updateControls() {
  const disabled=!state.record||state.busy||state.conflict;
  $('search').disabled=state.busy;$('filter').disabled=state.busy;
  document.querySelectorAll('.image-row').forEach(b=>b.disabled=state.busy);
  for(const id of ['prev','next','done','empty','defer','note','resume','export','imageSlider']) $(id).disabled=disabled;
  $('prev').disabled=disabled||state.index<=0;$('next').disabled=disabled||state.index>=state.items.length-1;
  $('undo').disabled=disabled||(!state.undo.length&&!state.draft.length);$('redo').disabled=disabled||!state.redo.length;
  $('delete').disabled=disabled||state.selected<0;
  $('clearAll').disabled=disabled||Boolean(state.drag)||(!state.record?.polygons.length&&!state.draft.length);
  $('resetStatus').disabled=disabled||Boolean(state.drag)||state.record?.status==='pending';
  $('finishPolygon').disabled=disabled||state.draft.length<3;
  $('cancelDraft').disabled=disabled||!state.draft.length;
  $('finishPolygon').hidden=state.draft.length<3;
  $('cancelDraft').hidden=!state.draft.length;
  document.querySelectorAll('[data-part]').forEach(b=>b.disabled=state.busy);
}
function svgElement(name,attributes) { const el=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attributes)) el.setAttribute(k,String(v));return el; }
function renderShapes() {
  $('shapes').replaceChildren();$('draftShape').replaceChildren();
  if(!state.record) return;
  const show=$('overlay').checked;
  if(show) state.record.polygons.forEach((savedPoints,index)=>{
    const points=index===state.selected&&state.drag?.type!=='rectangle'&&state.drag?.preview?state.drag.preview:savedPoints;
    const selected=index===state.selected;
    $('shapes').append(svgElement('polygon',{points:points.map(p=>p.join(',')).join(' '),class:`mask-polygon${selected?' selected':''}`,'data-polygon':index}));
    if(selected) points.forEach(([cx,cy],vertex)=>$('shapes').append(svgElement('circle',{cx,cy,r:5/state.scale,class:'handle','data-polygon':index,'data-vertex':vertex})));
  });
  if(state.draft.length) {
    $('draftShape').append(svgElement('polyline',{points:state.draft.map(p=>p.join(',')).join(' '),class:'draft-line'}));
    state.draft.forEach(([cx,cy])=>$('draftShape').append(svgElement('circle',{cx,cy,r:4/state.scale,class:'draft-point'})));
  }
  if(state.drag?.type==='rectangle'&&state.drag.preview) {
    $('draftShape').append(svgElement('polygon',{points:state.drag.preview.map(p=>p.join(',')).join(' '),class:'draft-line'}));
  }
  updateControls();
}
function renderRegions() {
  $('regionCount').textContent=state.record?.polygons.length||0;
  const buttons=(state.record?.polygons||[]).map((points,index)=>{
    const b=document.createElement('button');b.textContent=`${index+1} · ${points.length}점`;b.classList.toggle('selected',index===state.selected);
    b.onclick=()=>{if(state.busy)return;state.selected=index;setTool('select');renderRegions();renderShapes();};return b;
  });
  if(buttons.length) $('regionList').replaceChildren(...buttons);
  else {const span=document.createElement('span');span.className='muted';span.textContent='아직 그린 영역이 없습니다.';$('regionList').replaceChildren(span);}
}
function view() { $('viewport').setAttribute('transform',`translate(${state.offset.join(' ')}) scale(${state.scale})`);$('zoomText').textContent=`${Math.round(state.scale*100)}%`;renderShapes(); }
function fit() {
  if(!state.record)return;
  const box=$('stage').getBoundingClientRect();
  state.scale=Math.max(.1,Math.min((box.width-50)/state.record.width,(box.height-50)/state.record.height));
  state.offset=[(box.width-state.record.width*state.scale)/2,(box.height-state.record.height*state.scale)/2];view();
}
function coords(event, bounded=true) {
  const box=$('canvas').getBoundingClientRect();
  let x=(event.clientX-box.left-state.offset[0])/state.scale,y=(event.clientY-box.top-state.offset[1])/state.scale;
  if(bounded) {x=clamp(x,0,state.record.width);y=clamp(y,0,state.record.height);}
  return [x,y];
}
function setTool(tool) {
  if(state.drag)return;
  if(state.draft.length&&state.tool!==tool) {notice('그리던 영역을 완성하거나 Esc로 취소하고 도구를 바꿔 주세요.');return;}
  state.tool=tool;
  document.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('active',b.dataset.tool===tool));
  $('toolHint').textContent={quad:'사각형: 한 모서리에서 반대 모서리까지 드래그하세요. 생성 후 꼭짓점을 바로 조정할 수 있습니다.',polygon:'W 마지막 점 되돌리기 · E 다각형 확정 · A/D 이전/다음 · S 보류',select:'선택: 영역을 이동하거나 꼭짓점을 드래그해 수정하세요.'}[tool];
  $('canvas').style.cursor=tool==='select'?'default':'crosshair';renderShapes();
}
function finishDraft() {
  const error=polygonError(state.draft);
  if(error){notice(error);return;}
  remember();state.record.polygons.push(copy(state.draft));state.draft=[];
  state.selected=state.record.polygons.length-1;notice();geometryChanged();
}
function cancelDraft(){
  const drag=state.drag;state.drag=null;state.draft=[];
  if(drag&&$('canvas').hasPointerCapture(drag.pointerId))$('canvas').releasePointerCapture(drag.pointerId);
  notice();renderShapes();
}
function undo(redo=false) {
  if(!state.record||state.busy||state.conflict)return;
  if(state.draft.length&&!redo){state.draft.pop();renderShapes();return;}
  const from=redo?state.redo:state.undo,to=redo?state.undo:state.redo;
  if(!from.length)return;to.push(snapshot());const previous=from.pop();
  state.record.polygons=previous.polygons;state.record.status=previous.status;state.selected=-1;
  changed();renderRegions();renderShapes();
}
function removeSelected(){
  if(state.selected<0||state.busy||state.conflict)return;
  remember();state.record.polygons.splice(state.selected,1);state.selected=-1;geometryChanged();
}
function clearAll(){
  if(!state.record||state.busy||state.conflict||state.drag)return;
  if(!state.record.polygons.length){cancelDraft();return;}
  remember();state.record.polygons=[];state.draft=[];state.selected=-1;
  geometryChanged();notice('현재 이미지의 영역을 모두 지웠습니다. Ctrl+Z 또는 ↶로 복구할 수 있습니다.');
}
async function mark(status) {
  if(!requireNoDraft())return;
  if(status==='done'&&!state.record.polygons.length){notice('패널 영역을 먼저 그려 주세요. 패널이 없으면 ‘패널 없음’을 누르세요.');return;}
  if(status==='empty'&&state.record.polygons.length){notice('그린 영역이 있습니다. 영역을 삭제한 뒤 ‘패널 없음’을 선택하세요.');return;}
  remember();state.record.status=status;notice();changed();await flush();
  if(state.index<state.items.length-1) await loadImage(state.index+1);
  else notice('파트의 마지막 이미지입니다. 진행률과 보류 항목을 확인하고 ZIP을 내보내세요.');
}
async function resetStatus() {
  if(!requireNoDraft()||state.record.status==='pending')return;
  remember();state.record.status='pending';changed();await flush();
  notice('결정을 취소하고 미작업으로 되돌렸습니다. 그린 영역과 메모는 보존됩니다. Ctrl+Z로 복구할 수 있습니다.');
}

$('canvas').addEventListener('pointerdown',event=>{
  if(!state.record||state.busy||state.conflict||!state.imageReady||state.drag)return;
  if(event.button!==0&&event.button!==1)return;
  event.preventDefault();$('canvas').focus();
  if(event.button===1||state.space){state.drag={type:'pan',pointerId:event.pointerId,screen:[event.clientX,event.clientY],offset:[...state.offset]};$('canvas').setPointerCapture(event.pointerId);return;}
  const raw=coords(event,false);
  if(raw[0]<0||raw[1]<0||raw[0]>state.record.width||raw[1]>state.record.height)return;
  const point=coords(event);
  const editingHandle=event.target.dataset.vertex!==undefined&&!state.draft.length;
  if(state.tool==='quad'&&!editingHandle){
    state.drag={type:'rectangle',pointerId:event.pointerId,start:point,preview:rectangleFromDiagonal(point,point)};
    $('overlay').checked=true;notice();$('canvas').setPointerCapture(event.pointerId);renderShapes();return;
  }
  if(state.tool==='polygon'&&!editingHandle){
    if(state.draft.length>=100){notice('꼭짓점은 최대 100개입니다.');return;}
    state.draft.push(point);notice();renderShapes();
    return;
  }
  if(event.target.dataset.polygon!==undefined){
    state.selected=Number(event.target.dataset.polygon);
    state.drag={type:editingHandle?'vertex':'polygon',pointerId:event.pointerId,vertex:Number(event.target.dataset.vertex),start:point,original:copy(state.record.polygons[state.selected]),before:snapshot(),moved:false};
    $('canvas').setPointerCapture(event.pointerId);
  }else state.selected=-1;
  renderRegions();renderShapes();
});
$('canvas').addEventListener('pointermove',event=>{
  const drag=state.drag;if(!drag)return;
  if(drag.type==='pan'){state.offset=[drag.offset[0]+event.clientX-drag.screen[0],drag.offset[1]+event.clientY-drag.screen[1]];view();return;}
  const point=coords(event);drag.moved=true;
  if(drag.type==='rectangle')drag.preview=rectangleFromDiagonal(drag.start,point);
  else if(drag.type==='vertex'){drag.preview=copy(drag.original);drag.preview[drag.vertex]=point;}
  else drag.preview=translatePolygon(drag.original,point[0]-drag.start[0],point[1]-drag.start[1],state.record.width,state.record.height);
  renderShapes();
});
function endDrag(event) {
  const drag=state.drag;if(!drag)return;state.drag=null;
  if($('canvas').hasPointerCapture(event.pointerId))$('canvas').releasePointerCapture(event.pointerId);
  if(drag.type==='rectangle'){
    if(event.type==='pointercancel'||event.type==='lostpointercapture'){renderShapes();return;}
    const end=coords(event),points=rectangleFromDiagonal(drag.start,end);
    if(Math.abs(end[0]-drag.start[0])*state.scale<3||Math.abs(end[1]-drag.start[1])*state.scale<3){renderShapes();return;}
    const error=polygonError(points);if(error){notice(error);renderShapes();return;}
    remember();state.record.polygons.push(points);state.selected=state.record.polygons.length-1;notice();geometryChanged();return;
  }
  if(drag.type==='pan'||!drag.moved)return;
  const error=event.type==='pointercancel'||event.type==='lostpointercapture'?'영역 이동이 취소되었습니다.':polygonError(drag.preview);
  if(error){notice(error);renderShapes();return;}
  state.undo.push(drag.before);state.redo=[];state.record.polygons[state.selected]=drag.preview;notice();geometryChanged();
}
$('canvas').addEventListener('pointerup',endDrag);$('canvas').addEventListener('pointercancel',endDrag);
$('canvas').addEventListener('lostpointercapture',endDrag);
$('canvas').addEventListener('wheel',event=>{
  if(!state.record)return;event.preventDefault();if(state.drag)return;const point=coords(event,false),box=$('canvas').getBoundingClientRect();
  state.scale=clamp(state.scale*Math.exp(-event.deltaY*.0015),.2,24);
  state.offset=[event.clientX-box.left-point[0]*state.scale,event.clientY-box.top-point[1]*state.scale];view();
},{passive:false});
document.querySelectorAll('[data-tool]').forEach(b=>b.onclick=()=>setTool(b.dataset.tool));
$('fit').onclick=fit;$('overlay').onchange=renderShapes;$('undo').onclick=()=>undo();$('redo').onclick=()=>undo(true);
$('aiToggle').checked=storageGet('bitgaram-v2:ai-overlay')==='true';
$('aiToggle').onchange=()=>{storageSet('bitgaram-v2:ai-overlay',String($('aiToggle').checked));showPrediction();};
$('delete').onclick=removeSelected;$('clearAll').onclick=clearAll;$('finishPolygon').onclick=finishDraft;$('cancelDraft').onclick=cancelDraft;
$('prev').onclick=()=>run(()=>loadImage(state.index-1));$('next').onclick=()=>run(()=>loadImage(state.index+1));
$('imageSlider').oninput=()=>{
  $('position').textContent=`${$('imageSlider').value} / ${state.items.length}`;
  $('imageSlider').setAttribute('aria-valuetext',`${$('imageSlider').value} / ${state.items.length}번째 이미지`);
};
$('imageSlider').onchange=()=>run(async()=>{
  try { await loadImage(Number($('imageSlider').value)-1); }
  finally {
    $('imageSlider').value=state.index+1;
    $('position').textContent=`${state.index+1} / ${state.items.length}`;
    $('imageSlider').setAttribute('aria-valuetext',`${state.index+1} / ${state.items.length}번째 이미지`);
  }
});
$('done').onclick=()=>run(()=>mark('done'));$('empty').onclick=()=>run(()=>mark('empty'));$('defer').onclick=()=>run(()=>mark('deferred'));
$('resetStatus').onclick=()=>run(resetStatus);
$('imageList').onclick=event=>{const button=event.target.closest('[data-index]');if(button)run(()=>loadImage(Number(button.dataset.index)));};
$('filter').onchange=renderList;$('search').oninput=renderList;
$('resume').onclick=()=>run(async()=>{const index=state.items.findIndex(i=>!complete(i.status));if(index<0)notice('이 파트의 모든 이미지가 완료되었습니다.');else await loadImage(index);});
$('note').oninput=()=>{if(state.record){state.record.note=$('note').value;changed();}};
$('retry').onclick=()=>run(async()=>{notice();await flush();});
function download(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),30000);}
$('downloadDraft').onclick=()=>download(new Blob([JSON.stringify(state.recovery||state.record,null,2)],{type:'application/json'}),`${state.record.image_id}-recovery.json`);
$('reloadRecord').onclick=()=>run(async()=>{
  download(new Blob([JSON.stringify(state.recovery||state.record,null,2)],{type:'application/json'}),`${state.record.image_id}-recovery.json`);
  clearBackup();state.dirty=false;state.conflict=false;state.recovery=null;await loadImage(state.index);
  notice('복구용 초안을 내려받고 최신 저장본을 불러왔습니다.');
});
$('export').onclick=()=>run(async()=>{
  if(!requireNoDraft())return;await flush();
  const response=await fetch(`/api/export/${state.part}`);
  if(!response.ok)throw new Error((await response.json()).error);
  download(await response.blob(),`part-${state.part}.zip`);
  const remaining=state.items.filter(i=>!complete(i.status)).length;
  notice(remaining?`ZIP을 내보냈습니다. 미완료 ${remaining}장이 포함되어 있습니다. 미완료 영역은 초안 좌표만 포함되고 학습용 마스크에서는 제외됩니다.`:'모든 이미지가 완료된 ZIP을 내보냈습니다.');
});
document.addEventListener('keydown',event=>{
  if(event.target.closest('input,textarea,select')||event.altKey||!state.record)return;
  if(event.code==='Space'){event.preventDefault();state.space=true;return;}
  if(event.key==='Escape'&&state.drag){event.preventDefault();cancelDraft();return;}
  if(state.busy||state.conflict||state.drag)return;
  const key=/^Key[A-Z]$/.test(event.code)?event.code.slice(3).toLowerCase():event.key.toLowerCase();
  if((event.ctrlKey||event.metaKey)&&key==='z'){event.preventDefault();undo(event.shiftKey);return;}
  if(event.ctrlKey||event.metaKey)return;
  const actions={q:()=>setTool('quad'),p:()=>setTool('polygon'),v:()=>setTool('select'),f:fit,escape:cancelDraft,delete:removeSelected,
    w:()=>undo(),e:()=>{if(state.draft.length)finishDraft();},
    enter:()=>{if(state.draft.length)finishDraft();},a:()=>run(()=>loadImage(state.index-1)),d:()=>run(()=>loadImage(state.index+1)),
    s:()=>run(()=>mark('deferred')),x:()=>run(()=>mark('empty')),r:()=>run(()=>mark('done')),t:()=>run(resetStatus)};
  if(actions[key]){event.preventDefault();actions[key]();}
});
document.addEventListener('keyup',event=>{if(event.code==='Space')state.space=false;});
window.addEventListener('blur',()=>{state.space=false;if(state.drag)cancelDraft();});
window.addEventListener('beforeunload',event=>{if(state.dirty||state.draft.length||state.saving||state.drag){event.preventDefault();event.returnValue='';}});
// Layout changes (draft buttons, notices, AI status) must not reset the user's view.
// Fit only when loading an image or explicitly choosing 화면 맞춤 / F.

async function init() {
  state.config=await api('/api/config');
  state.config.parts.forEach(({part,total})=>{
    const b=document.createElement('button');b.dataset.part=part;b.textContent=part;b.title=`파트 ${part} · ${total}장`;b.setAttribute('aria-label',`파트 ${part} (${total}장)`);
    b.onclick=()=>run(()=>selectPart(part));$('parts').append(b);
  });
  updateControls();
  const last=Number(storageGet('bitgaram-v2:part'));
  if(last>=1&&last<=4)await run(()=>selectPart(last));
}
init().catch(error=>notice(`시작하지 못했습니다: ${error.message}`));
