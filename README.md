# SolarCast

SolarCast의 최종 목표는 국내 여러 지역에서 지도 위치를 선택하면 최신 이용 가능 항공사진을 조회하고, 팀이 학습한 모델을 실행해 옥상 태양광 설치 가능 면적과 예상 연간 발전량을 보여주는 것입니다. 빛가람동은 기존 연구 사례이지 서비스의 고정 분석 범위가 아닙니다.

## 서비스 실행

Task 1 웹은 지도와 조작·결과 화면 시안입니다. 실제 분석을 수행하지 않습니다. 자세한 지도 설정과 Vercel 제출 절차는 [웹서비스 안내](5_web_service/README.md)를 보세요.

```bash
cd 5_web_service
npm ci
npm run dev
```

프로덕션 빌드:

```bash
npm run build
```

## 최종 분석 흐름

지도에서 구역 선택 → 분석용 영상·건물 조회 → 옥상/장애물 분할 및 기존 패널 탐지 → 설치 가능 영역과 연간 발전량 산정 → 지도와 건물별 결과 표시. 기존 연구 코드는 웹 시안과 독립적으로 보관합니다.

## 프로젝트 문서와 구성

- [PRD](PRD.md): 서비스 요구사항, 사용자 흐름, 기능 범위와 배포 계획
- [모델 명세](MODEL_SPEC.md): 데이터, NLR 모델, 학습·평가 및 추론 계획
- [웹서비스 안내](5_web_service/README.md): 실행, 화면 시안 범위, 지도 설정과 제출 절차
- [항공영상 준비](1_aerial_imagery/README.md)
- [옥상 설치가능 영역 분석](2_roof_detection/README.md)
- [태양광 패널 탐지 및 학습](3_panel_detection/README.md)
- [그림자·발전량 시뮬레이션](4_shadow_simulation/README.md)

| 경로 | 역할 |
|---|---|
| `1_aerial_imagery/` | 항공영상 수집과 타일 처리 |
| `2_roof_detection/` | 옥상 설치가능 영역 분석 |
| `3_panel_detection/` | 태양광 패널 라벨링, 탐지, 학습 |
| `4_shadow_simulation/` | 그림자와 연간 발전량 계산 |
| `5_web_service/` | SolarCast 웹 애플리케이션 |
| `data/`, `outputs/` | 공간 데이터 및 파이프라인 산출물 |

## 데이터 이용 안내

지도 배경과 학습 데이터의 이용 조건은 제공처별로 확인해야 합니다. 웹은 VWorld WMTS로 배경 타일을 조회하며, 원본·축소 타일과 사전 계산 결과를 빌드에 포함하지 않습니다. 연구용 축소 타일이 필요하면 [항공영상 모듈 실행 안내](1_aerial_imagery/README.md#실행-예시)를 참고하고, 허가 없이 원본 영상을 재배포하지 마세요.
