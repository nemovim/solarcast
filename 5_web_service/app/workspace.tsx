'use client';

import { useState, type CSSProperties } from 'react';
import { ArrowDownRight, ArrowUpRight, CalendarDays, CircleHelp, Crosshair, Layers3, MapPin, ScanLine, Sun, SunDim } from 'lucide-react';
import MapPanel, { PLACES, type MapPosition } from './map-panel';
import { DEFAULT_LAYERS, LAYERS } from './demo-building';

export default function Workspace({ vworldEnabled }: { vworldEnabled: boolean }) {
  const [target, setTarget] = useState<MapPosition>(PLACES[0]);
  const [position, setPosition] = useState<MapPosition>(PLACES[0]);
  const [radius, setRadius] = useState(500);
  const [date, setDate] = useState('2026-06-21');
  const [notice, setNotice] = useState(false);
  const [visibility, setVisibility] = useState(DEFAULT_LAYERS);

  return (
    <>
      <a className="skip-link" href="#workspace">본문으로 건너뛰기</a>
      <header className="masthead">
        <div className="shell flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="brand-symbol"><Sun size={23} strokeWidth={2.3} aria-hidden="true" /></div>
            <div><div className="brand" translate="no">SolarCast<span>.</span></div><div className="brand-sub">옥상 태양광 분석</div></div>
          </div>
          <span className="status"><i />Task 1 시안</span>
        </div>
      </header>

      <main id="workspace" className="shell workspace">
        <div className="page-intro flex items-end justify-between gap-4">
          <div>
            <h1>우리 동네 태양광 잠재력</h1>
            <p>지도에서 위치를 정하고, 3D 건물 시연으로 옥상 분석 결과를 살펴보세요.</p>
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
            <MapPanel target={target} onPosition={setPosition} radius={radius} visibility={visibility} vworldEnabled={vworldEnabled} />
            <div className="map-footer flex flex-wrap items-center justify-between gap-2">
              <span className="flex items-center gap-2"><Crosshair size={15} aria-hidden="true" /> 지도 중심</span>
              <span className="coords">{position.lat.toFixed(5)}° N <b aria-hidden="true">/</b> {position.lng.toFixed(5)}° E</span>
            </div>
          </section>

          <section className="panel settings min-w-0" aria-labelledby="settings-heading">
            <div className="module-heading flex items-center justify-between gap-2">
              <h2 id="settings-heading" className="flex items-center gap-2"><ScanLine size={18} aria-hidden="true" /> 분석 설정</h2>
              <span className="sub-status">조작 시안</span>
            </div>

            <div className="settings-body">
              <fieldset>
                <legend>분석 범위 <span>· 중심 기준</span></legend>
                <div className="segmented grid grid-cols-3">
                  {[250, 500, 1000].map((value) => (
                    <label key={value}><input className="sr-only" type="radio" name="radius" checked={radius === value} onChange={() => setRadius(value)} /><span>{value === 1000 ? '1 km' : `${value} m`}</span></label>
                  ))}
                </div>
              </fieldset>

              <fieldset className="time-settings">
                <legend className="flex items-center gap-1.5"><CalendarDays size={15} aria-hidden="true" /> 일조 기준 시점 <span>· 한국 시간</span></legend>
                <div className="date-fields grid gap-2">
                  <input type="date" aria-label="분석 날짜" name="date" value={date} onChange={(event) => setDate(event.target.value)} autoComplete="off" />
                  <input type="time" aria-label="분석 시각" name="time" defaultValue="12:00" step={300} autoComplete="off" />
                </div>
                <div className="preset-links flex flex-wrap gap-1.5">
                  {[
                    { date: '2026-03-20', label: '춘분' }, { date: '2026-06-21', label: '하지' },
                    { date: '2026-09-23', label: '추분' }, { date: '2026-12-22', label: '동지' },
                  ].map((preset) => <button key={preset.date} type="button" onClick={() => setDate(preset.date)}>{preset.label}</button>)}
                </div>
              </fieldset>

              <fieldset className="layer-settings">
                <legend className="flex items-center gap-1.5"><Layers3 size={16} aria-hidden="true" /> 지도 레이어 <span>· 옥상 레이어는 시연용</span></legend>
                <div className="layer-grid grid grid-cols-2 gap-2">
                  {LAYERS.map((layer) => <label key={layer.id} className="layer-option" style={{ '--layer-color': layer.color } as CSSProperties}>
                    <input type="checkbox" name={layer.id} checked={visibility[layer.id]} onChange={(event) => { const checked = event.currentTarget.checked; setVisibility((previous) => ({ ...previous, [layer.id]: checked })); }} /> {layer.name}
                  </label>)}
                </div>
              </fieldset>

              <div className="start-area">
                <button type="button" className="start-button flex w-full items-center justify-between gap-2" onClick={() => setNotice(true)}>
                  <span className="flex items-center gap-2"><SunDim size={20} aria-hidden="true" /> 분석 시작</span><ArrowUpRight size={18} aria-hidden="true" />
                </button>
                <p role="status" className={notice ? 'notice-active' : ''}>{notice ? '분석은 아직 미구현이에요. 실제 요청을 보내지 않았어요.' : '분석용 영상 처리와 AI 추론은 아직 연결되지 않았어요.'}</p>
              </div>
            </div>
          </section>
        </div>

        <footer className="workspace-footer flex flex-wrap justify-between gap-x-4 gap-y-2" id="analysis-note" tabIndex={-1}>
          <p><CircleHelp size={15} aria-hidden="true" /> 화면 시안입니다. 실제 분석이나 시공 판단에 사용할 수 없어요.</p>
          <p>항공사진 조회 → AI 추론 → 면적·발전량 추정 <span>연동 예정</span></p>
        </footer>
      </main>
    </>
  );
}
