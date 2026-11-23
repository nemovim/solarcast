# 표준 산출물 통합 저장소 (Standardized Outputs)

각 파이프라인 모듈(1~4단계)이 생성한 최종 결과물을 한곳에 모아 관리하는 표준 디렉토리입니다.
웹서비스(`5_web_service`)는 이 폴더를 단일 진입점으로 참조하여 시연 대시보드에 표출합니다.

---

### 모듈별 최종 산출물 목록

| 모듈 | 파일명 | 데이터 형식 | 설명 |
|---|---|---|---|
| **1단계**<br>항공 영상 | `1_aerial/buildings.geojson` | GeoJSON (Polygon) | 나주 혁신도시(빛가람동) VWorld 건물 외곽선 2,207동 및 층수/높이 속성 |
| | `1_aerial/qc_report.csv` | CSV | 타일 3,410장의 선명도(라플라시안 분산) 및 품질 메타데이터 |
| **2단계**<br>옥상 탐지 | `2_roof/available_roof.geojson` | GeoJSON (Polygon) | SAM ViT-H 기반 옥상 유휴부지 정밀 분할 폴리곤 |
| | `2_roof/panel_layout.geojson` | GeoJSON (Polygon) | 옥상 유휴부지 내 400W(1m×2m) 태양광 모듈 자동 격자 배치 레이아웃 (49,863장) |
| | `2_roof/rooftop.geojson` | GeoJSON (Polygon) | 건물별 건축물대장 및 옥상 면적 인계 속성 (737동) |
| **3단계**<br>패널 탐지 | `3_panels/panels_final.geojson` | GeoJSON (MultiPolygon) | **[최종 채택]** AI-Hub 파인튜닝 NLR 모델 탐지 결과 (conf 0.20, 1,049개) |
| | `3_panels/panels_masking_v2.geojson` | GeoJSON (MultiPolygon) | [비교 레이어] 4인 마스킹 v2 full 파인튜닝 NLR 모델 결과 (conf 0.15, 1,379개) |
| | `3_panels/panels_overfit.geojson` | GeoJSON (MultiPolygon) | [비교 레이어] 과적합 진단 NLR 모델 결과 (conf 0.10, 2,476개) |
| | `3_panels/panels_v3.geojson` | GeoJSON (MultiPolygon) | [비교 레이어] 기본 파인튜닝 v3 모델 결과 (695개) |
| **4단계**<br>시뮬레이션 | `4_simulation/shadow_11-11.json` | JSON | 발표일(11-11) 전역 5분 간격 건물 3D 그림자 음영률 & NASA 일사량 기반 연간 발전량 캐시 |

---

### 공통 공간 좌표계
* 모든 GeoJSON 파일은 **WGS84 (EPSG:4326) 경위도 좌표계**로 표준화되어 있습니다.
