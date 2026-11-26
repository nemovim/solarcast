# 웹 프런트엔드

이 폴더에서 `npm ci`로 의존성을 설치하고 `npm run dev`로 실행한다. 실행 스크립트가 저장소 루트 `.env`의 `VWORLD_KEY`와 `VWORLD_DOMAIN`을 읽는다. 별도 Python 서버나 브라우저 모델 추론 없이 사전 추론 결과와 건물 외곽선으로 패널 표시·주변 건물 그림자와 발전량을 추정한다. 지도와 NASA POWER 조회는 Next.js API 경유다.

지도 화면에서 선택한 패널·옥상 모델이 모두 처리한 부분만 분석한다. 노란 선은 두 모델 모두 처리한 타일을 나타내며 미분석 부분은 합계에서 제외한다. `public/precomputed/panel/`과 `public/precomputed/roof/`의 모델별 결과와 각 `index.json`은 GPU 추론 후 함께 반영해야 한다. `public/maplibre-gl-worker.mjs`는 MapLibre 패키지 버전과 일치시켜야 한다.

Vercel Root Directory는 `frontend`이며 VWorld 환경변수를 Vercel 프로젝트에도 설정해야 한다. 옥상을 평지붕으로 가정하며 장애물과 나무는 반영하지 않는다. 높이가 없는 건물은 15m, 층수만 있는 건물은 층당 3.3m로 추정한다. 초기 가정과 추론 결과는 실측 검증이 필요하다. 이번 변경은 배포하지 않는다.
