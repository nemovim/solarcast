# SolarCast 웹 · Task 1

옥상 태양광 설치 후보 면적과 예상 연간 발전량을 보여줄 서비스의 **화면 시안**이다. 지도 탐색과 도시 바로가기는 동작한다. 지도 위 범위 원은 설정을 표현한 도식이지 실제 반경 도형이 아니다. **3D 건물 시연**은 가상 건물로 이동해 지도 위 결과 팝업을 열고, 건물을 눌러 다시 열 수 있다. 건물·옥상 음영·기존 패널·설치 가능 영역 토글은 시연 데이터에만 적용된다. **분석 시작**은 미구현을 안내하며, 영상 수집, 추론, 계산, 분석 이력 저장은 없다.

## 로컬 실행

Node.js 22 이상을 사용한다. 저장소 루트에서:

```sh
cd 5_web_service
npm ci
cp .env.example .env.local  # 선택: 브이월드 항공사진·건물 표시 설정
npm run dev
```

서버 환경 변수 `VWORLD_KEY`에 VWorld 키를 설정한다. URL 제한이 있는 키는 `VWORLD_DOMAIN`에 해당 키에 등록된 서비스 URL을 지정한다. 키는 브라우저로 보내지 않는다. `Satellite` WMTS는 256픽셀의 z19 원본까지만 요청하고 그 이상은 확대 표시한다. 실제 건물은 충분히 확대하면 현재 화면 범위의 `LT_C_SPBD` GeoJSON을 조회한다. 공급 데이터의 높이, 층수 × 3.3m, 기본 높이 15m 순으로 표시 높이를 결정하며 팝업에서 출처를 구분한다. 이 팝업에는 실제 분석 수치가 나오지 않는다.

키가 없거나 항공사진 초기 로딩이 실패하면 OpenStreetMap 기본 지도를 표시한다. OSM은 소규모 시연용이며 [타일 이용 정책](https://operations.osmfoundation.org/policies/tiles/)을 따른다. 공개 운영 전 VWorld 이용 조건 및 서버 중계·요청량 제한을 확인한다.

```sh
npm run typecheck
npm run build
```

## 분석 연결 지점

`app/workspace.tsx`의 분석 시작은 로컬 안내 상태만 바꾸며 네트워크 호출이 없다. `app/map-panel.tsx`는 지도 중심과 확대 수준을 화면에 전달한다. 최종 연동 시 지도 중심·반경·선택 시각을 [요청 계약](../MODEL_SPEC.md#5-시스템-구조와-인터페이스)에 연결하고, 상태/결과를 결과 패널과 지도 레이어에 전달한다. 현재 예시는 프로덕션 결과 타입이나 응답으로 사용하지 않는다.

기존 연구용 건물, 옥상·배치, 패널 탐지 자료는 각각 `outputs/1_aerial/`, `outputs/2_roof/`, `outputs/3_panels/`에 있다. [항공영상 타일 생성](../1_aerial_imagery/README.md#실행-예시)과 [그림자 사전 계산](../4_shadow_simulation/README.md#실행-방법)은 연구 환경에서 별도로 실행한다. 이 경로와 산출물은 현재 웹의 빌드나 실행에 필요하지 않다.

## Vercel 과제 제출

1. 제출용 GitHub 저장소에 웹 소스와 [PRD](../PRD.md), [모델 명세](../MODEL_SPEC.md)를 포함한다. 공개 전 원본 영상·라벨·토큰의 재배포 가능 여부를 확인한다.
2. Vercel에서 저장소를 가져오고 Root Directory를 `5_web_service`, Framework Preset을 **Next.js**로 설정한다. 기존 Vite의 `dist` 설정이 남아 있으면 제거해 Output Directory를 기본값으로 돌린다.
3. VWorld 항공사진과 건물을 표시하려면 해당 프로젝트 환경 변수에 `VWORLD_KEY`와 필요한 경우 `VWORLD_DOMAIN`을 등록한 뒤 배포한다.
4. 로그인하지 않은 브라우저에서 GitHub 문서와 사이트가 열리는지 확인하고, 두 URL을 제출한다. 팀원 이름·학번을 포함한 2–3쪽 제안 보고서 PDF는 별도 제출물이다.
