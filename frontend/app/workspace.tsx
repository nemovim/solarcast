'use client';

import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react';
import { ArrowDownRight, ArrowUpRight, CalendarDays, CircleHelp, Crosshair, Layers3, MapPin, ScanLine, Sun, SunDim } from 'lucide-react';
import MapPanel, { INITIAL_POSITION, PLACES, type MapPosition, type Viewport } from './map-panel';
import { DEFAULT_LAYERS, LAYERS } from './demo-building';
import type { AnalysisResult } from './analysis';
import type { FeatureCollection, Polygon } from 'geojson';
import { coverageGeometry, loadAvailable, readManifest, readModels, type PrecomputedModel } from './client-precomputed';
import { estimateSolar } from './client-solar';

export default function Workspace({ vworldEnabled }: { vworldEnabled: boolean }) {
  const [target, setTarget] = useState<MapPosition>(INITIAL_POSITION);
  const [position, setPosition] = useState<MapPosition>(INITIAL_POSITION);
  const [viewport, setViewport] = useState<Viewport | null>(null);
  const onViewport = useCallback((next: Viewport) => setViewport((previous) =>
    previous && previous.every((point, i) => point.every((value, j) => Math.abs(value-next[i][j]) < 1e-8)) ? previous : next), []);
  const [date, setDate] = useState('2026-06-21');
  const [time, setTime] = useState('12:00');
  const [panelThreshold, setPanelThreshold] = useState(0.5);
  const [models, setModels] = useState<PrecomputedModel[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [roofModels, setRoofModels] = useState<PrecomputedModel[]>([]);
  const [selectedRoofModel, setSelectedRoofModel] = useState('');
  const [modelError, setModelError] = useState('');
  const [coverage, setCoverage] = useState<FeatureCollection<Polygon>>({type:'FeatureCollection',features:[]});
  const [notice, setNotice] = useState('분석할 위치와 범위를 선택하세요.');
  const [busy, setBusy] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [resultSettings, setResultSettings] = useState<string | null>(null);
  const currentSettings = JSON.stringify({ viewport, date, time, panelThreshold, selectedModel, selectedRoofModel });
  const activeRequest = useRef(0);
  const [visibility, setVisibility] = useState(DEFAULT_LAYERS);
  useEffect(() => () => { activeRequest.current++; }, []);
  useEffect(() => {
    let cancelled = false;
    Promise.all([readModels('panel'), readModels('roof')]).then(([available, roofs]) => {
      if (cancelled) return;
      setModels(available);
      setSelectedModel(available[0]?.id ?? '');
      setRoofModels(roofs);
      setSelectedRoofModel(roofs[0]?.id ?? '');
    }).catch((error) => { if (!cancelled) setModelError(error instanceof Error ? error.message : '모델 목록을 불러오지 못했습니다.'); });
    return () => { cancelled = true; };
  }, []);
  useEffect(() => {
    let cancelled = false;
    setCoverage({type:'FeatureCollection',features:[]});
    setModelError('');
    if (selectedModel && selectedRoofModel) void Promise.all([readManifest('panel', selectedModel), readManifest('roof', selectedRoofModel)]).then(([panel, roof]) => {
      if (!cancelled) setCoverage(coverageGeometry(panel, roof));
    }).catch((error) => { if (!cancelled) setModelError(error instanceof Error ? error.message : '분석 가능 영역을 불러오지 못했습니다.'); });
    return () => { cancelled = true; };
  }, [selectedModel, selectedRoofModel]);
  async function startAnalysis() {
    const sequence = ++activeRequest.current;
    const requestedSettings = currentSettings;
    setBusy(true); setNotice('사전 추론 결과를 확인하는 중입니다.');
    try {
      if (!date || !time) throw new Error('날짜와 시각을 선택해 주세요.');
      if (!viewport) throw new Error('지도 화면이 준비될 때까지 기다려 주세요.');
      if (!selectedModel) throw new Error('사용할 사전 추론 모델이 없습니다.');
      if (!selectedRoofModel) throw new Error('사용할 사전 추론 옥상 모델이 없습니다.');
      const progress = (message: string) => { if (sequence === activeRequest.current) setNotice(message); };
      const { panels, roofs, panelManifest: manifest, roofManifest, coverage: analyzedCoverage, processed, partial } = await loadAvailable(selectedModel,selectedRoofModel,viewport,panelThreshold,progress);
      const result = await estimateSolar(viewport,panels,roofs,`${date}T${time}:00+09:00`,progress,analyzedCoverage);
      if (sequence !== activeRequest.current) return;
      result.models.panel=manifest.model;
      result.models.roof=roofManifest.model;
      result.warnings.unshift(`패널은 사전 추론 결과입니다. 영상 촬영일 ${manifest.imagery_observed_at ?? '미상'} · 결과 갱신 ${manifest.updated_at?.slice(0,10) ?? '미상'}.`);
      if (partial) result.warnings.unshift(`화면 중 두 모델의 결과가 있는 ${processed}개 타일만 분석했습니다. 미분석 영역은 합계에 포함되지 않습니다.`);
      setAnalysis(result); setResultSettings(requestedSettings);
      setNotice(partial ? `화면 일부를 분석했습니다. (${processed}개 타일)` : '패널과 옥상의 사전 추론 결과를 적용했습니다.');
    } catch (error) {
      if (sequence === activeRequest.current) setNotice(error instanceof Error ? error.message : '분석에 실패했습니다.');
    } finally { if (sequence === activeRequest.current) setBusy(false); }
  }

  return (
    <>
      <a className="skip-link" href="#workspace">본문으로 건너뛰기</a>
      <header className="masthead">
        <div className="shell flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="brand-symbol"><Sun size={23} strokeWidth={2.3} aria-hidden="true" /></div>
            <div><div className="brand" translate="no">SolarCast<span>.</span></div><div className="brand-sub">옥상 태양광 분석</div></div>
          </div>
          <span className="status"><i />분석 서비스</span>
        </div>
      </header>

      <main id="workspace" className="shell workspace">
        <div className="page-intro flex items-end justify-between gap-4">
          <div>
            <h1>우리 동네 태양광 잠재력</h1>
            <p>지도에서 위치를 정하고 옥상 태양광 잠재량을 분석하세요.</p>
          </div>
          <a className="help-link shrink-0" href="#analysis-note"><CircleHelp size={16} aria-hidden="true" /> 시안 안내 <ArrowDownRight size={14} aria-hidden="true" /></a>
        </div>

        <div className="workspace-grid grid gap-4">
          <section className="panel map-card min-w-0" aria-labelledby="map-heading">
            <div className="module-heading flex flex-wrap items-center justify-between gap-3">
              <h2 id="map-heading" className="flex items-center gap-2"><MapPin size={18} aria-hidden="true" /> 분석 위치</h2>
              <div className="places flex flex-wrap gap-1.5" aria-label="도시 바로가기">
                {PLACES.map((place) => <button type="button" key={place.name} onClick={() => { setTarget({ ...place }); setPosition(place); }}>{place.name}</button>)}
              </div>
            </div>
            <MapPanel target={target} onPosition={setPosition} onViewport={onViewport} visibility={visibility} vworldEnabled={vworldEnabled} analysis={analysis} coverage={coverage} />
            <div className="map-footer flex flex-wrap items-center justify-between gap-2">
              <span className="flex items-center gap-2"><Crosshair size={15} aria-hidden="true" /> 지도 중심</span>
              <span className="coords">{position.lat.toFixed(5)}° N <b aria-hidden="true">/</b> {position.lng.toFixed(5)}° E</span>
            </div>
          </section>

          <section className="panel settings min-w-0" aria-labelledby="settings-heading">
            <div className="module-heading flex items-center justify-between gap-2">
              <h2 id="settings-heading" className="flex items-center gap-2"><ScanLine size={18} aria-hidden="true" /> 분석 설정</h2>
                <span className="sub-status">분석 설정</span>
            </div>

            <div className="settings-body">
              <fieldset><legend>분석 범위</legend><p>화면에서 노란색으로 표시된 구역만 분석합니다. 노란 구역은 선택한 패널·옥상 모델이 모두 처리한 영역입니다.</p><p>미분석 영역은 결과와 합계에 포함되지 않습니다.</p></fieldset>

              <fieldset>
                <legend><label htmlFor="panel-model">패널 탐지 모델</label></legend>
                <select id="panel-model" value={selectedModel} disabled={busy || !models.length} onChange={(event) => setSelectedModel(event.target.value)}>
                  {models.length ? models.map((model) => <option key={model.id} value={model.id}>{model.name === model.id ? model.name : `${model.name} (${model.id})`}</option>) : <option value="">사전 추론 결과 없음</option>}
                </select>
                {modelError && <p role="alert">{modelError}</p>}
              </fieldset>

              <fieldset>
                <legend><label htmlFor="roof-model">옥상 탐지 모델</label></legend>
                <select id="roof-model" value={selectedRoofModel} disabled={busy || !roofModels.length} onChange={(event) => setSelectedRoofModel(event.target.value)}>
                  {roofModels.length ? roofModels.map((model) => <option key={model.id} value={model.id}>{model.name === model.id ? model.name : `${model.name} (${model.id})`}</option>) : <option value="">사전 추론 결과 없음</option>}
                </select>
              </fieldset>

              <fieldset className="time-settings">
                <legend className="flex items-center gap-1.5"><CalendarDays size={15} aria-hidden="true" /> 일조 기준 시점 <span>· 한국 시간</span></legend>
                <div className="date-fields grid gap-2">
                  <input type="date" aria-label="분석 날짜" name="date" value={date} onChange={(event) => setDate(event.target.value)} autoComplete="off" />
                  <input type="time" aria-label="분석 시각" name="time" value={time} onChange={(event) => setTime(event.target.value)} step={300} autoComplete="off" />
                </div>
                <div className="preset-links flex flex-wrap gap-1.5">
                  {[
                    { date: '2026-03-20', label: '춘분' }, { date: '2026-06-21', label: '하지' },
                    { date: '2026-09-23', label: '추분' }, { date: '2026-12-22', label: '동지' },
                  ].map((preset) => <button key={preset.date} type="button" onClick={() => setDate(preset.date)}>{preset.label}</button>)}
                </div>
              </fieldset>

              <fieldset className="layer-settings">
                <legend className="flex items-center gap-1.5"><Layers3 size={16} aria-hidden="true" /> 지도 레이어 <span>· 분석 완료 후 결과 표시</span></legend>
                <div className="layer-grid grid grid-cols-2 gap-2">
                  {LAYERS.map((layer) => <label key={layer.id} className="layer-option" style={{ '--layer-color': layer.color } as CSSProperties}>
                    <input type="checkbox" name={layer.id} checked={visibility[layer.id]} onChange={(event) => { const checked = event.currentTarget.checked; setVisibility((previous) => ({ ...previous, [layer.id]: checked })); }} /> {layer.name}
                  </label>)}
                </div>
              </fieldset>

              <fieldset>
                <legend>패널 탐지 신뢰도 <span>· {(panelThreshold * 100).toFixed(0)}%</span></legend>
                <input type="range" min="0.05" max="0.95" step="0.05" value={panelThreshold} onChange={(event) => setPanelThreshold(Number(event.target.value))} aria-label="패널 탐지 신뢰도" />
                <p>변경 후 분석을 다시 시작하면 설치 후보와 발전량에도 반영됩니다.</p>
              </fieldset>

              <div className="start-area">
                <button type="button" className="start-button flex w-full items-center justify-between gap-2" disabled={busy} onClick={startAnalysis}>
                  <span className="flex items-center gap-2"><SunDim size={20} aria-hidden="true" /> 분석 시작</span><ArrowUpRight size={18} aria-hidden="true" />
                </button>
                <p role="status" className="notice-active">{notice}</p>
                {analysis && resultSettings !== currentSettings && <p>표시된 결과는 이전 화면 범위와 설정을 기준으로 합니다. 현재 설정을 적용하려면 다시 분석해 주세요.</p>}
                {analysis && <div><h3>분석 구역 합계</h3>
                  <p>{analysis.totals.availableAreaM2 == null ? '설치 후보 미계산' : `설치 후보 ${analysis.totals.availableAreaM2.toLocaleString()}㎡`} · {analysis.totals.capacityKw == null ? '용량 미계산' : `${analysis.totals.capacityKw.toLocaleString()} kW`} · {analysis.totals.annualEnergyKwh == null ? '발전량 미계산' : `${analysis.totals.annualEnergyKwh.toLocaleString()} kWh/년`}</p>
                  <h3>건물별 분석 결과</h3>
                  {analysis.warnings.map((warning, index) => <p key={index}>{warning}</p>)}
                  {analysis.summaries.length === 0 ? <p>분석 대상 건물이 없습니다.</p> : analysis.summaries.map((building) => <p key={building.buildingId}>{building.name} · {building.availableAreaM2 == null ? '면적 미계산' : `${building.availableAreaM2.toLocaleString()}㎡`} · {building.capacityKw == null ? '용량 미계산' : `${building.capacityKw.toLocaleString()} kW`} · {building.annualEnergyKwh == null ? '발전량 미계산' : `${building.annualEnergyKwh.toLocaleString()} kWh/년`}</p>)}
                </div>}
              </div>
            </div>
          </section>
        </div>

        <footer className="workspace-footer flex flex-wrap justify-between gap-x-4 gap-y-2" id="analysis-note" tabIndex={-1}>
          <p><CircleHelp size={15} aria-hidden="true" /> 분석 결과는 추정치이며 시공 판단에는 현장 검토가 필요합니다.</p>
          <p>사전 추론 결과 → 면적·그림자·발전량 추정</p>
        </footer>
      </main>
    </>
  );
}
