# SolarCast Version 2

VWorld 건물 외곽선과 사전 추론한 패널·옥상 위치로 태양광 잠재량을 추정하는 개발용 MVP입니다. 제출한 [PRD](docs/PRD.md)와 [모델 명세](docs/MODEL_SPEC.md)는 제출 당시 문서로 보존합니다.

## 실행

1. 루트 `.env.example`을 `.env`로 복사하고 `VWORLD_KEY`와 필요한 경우 `VWORLD_DOMAIN`을 입력합니다.
2. `frontend` 폴더에서 `npm ci`를 실행한 후 `npm run dev`로 웹을 실행합니다. Python 서버는 필요하지 않습니다. 웹 실행 스크립트가 루트 `.env`를 읽습니다.
3. 브라우저에서 `http://localhost:5173`을 열고 분석할 지역이 화면에 들어오도록 지도를 이동·확대한 뒤 시각을 선택합니다. 화면이 너무 넓으면 분석을 위해 확대해야 합니다. 웹의 가상 건물 예시는 실제 결과와 구분됩니다.

웹은 `frontend/public/precomputed/`의 결과를 읽고 두 모델의 처리가 모두 완료된 영역만 분석합니다. 미처리 영역은 합계에서 제외합니다. Vercel 배포 시 프로젝트 루트는 `frontend`이며 VWorld 환경변수를 Vercel에 별도로 등록합니다.

## GPU 사전 추론

Backend.AI GPU 환경의 저장소 루트에서 준비하고 실행합니다.

1. CUDA PyTorch가 설치된 GPU 환경에서 `python -m pip install -r requirements.txt`로 기본 의존성을 설치합니다. 이어서 `python -m pip install openmim mmengine==0.10.7 mmdet==3.3.0`과 `mim install mmcv==2.1.0`으로 모델 의존성을 준비합니다. MMCV는 설치된 PyTorch·CUDA와 맞는 빌드가 필요하며 GPU 환경에 따라 별도 빌드가 필요할 수 있습니다. 이 조합은 과거 패널 연구의 버전이며 현재 GPU 환경에서는 아직 검증하지 않았습니다.
   Shapely 2.x가 필요하며 affine 3.x는 미검증 상태입니다. 설치 후 `python -m pip check`로 의존성 충돌을 확인하세요.
2. 루트 `.env`에 `VWORLD_KEY`와 필요한 `VWORLD_DOMAIN`을 설정합니다.
3. 패널 가중치를 `models/panel/`에, SAM ViT-H 옥상 가중치를 `models/roof/`에 둡니다. 기본 파일은 각각 `best_model.pth`, `model.pth`입니다. 패널 모델 설정은 `modules/panel/configs/inference.py`이며 가중치와 호환되어야 합니다. 옥상 어댑터는 기존 SAM ViT-H 구조를 사용합니다.
4. 아래 명령으로 사전 추론을 실행합니다. 빛가람 기본 범위에는 별도 타일 목록이 필요하지 않습니다.

```bash
python -m pipelines.precompute
```

전체 실행 전에 다음 명령으로 한 타일을 검증할 수 있습니다. 두 모델이 해당 GPU에서 실제 추론을 수행합니다.

```bash
python -m pipelines.precompute --bounds 446793 207630 446793 207630
```

웹 배포 빌드는 `frontend`에서 `npm run build`로 실행합니다. 이 명령은 웹팩을 사용합니다.

다른 가중치를 선택하려면 이름을 옵션으로 적습니다. 예를 들어 `models/panel/my_panel.pth`와 `models/roof/my_roof.pth`는 아래처럼 지정합니다. `.pth`는 생략할 수 있습니다.

```bash
python -m pipelines.precompute --panel my_panel --roof my_roof
```

패널의 다른 모델 설정은 `--panel-config <경로>`로 지정합니다. 가중치 파일 이름은 영문, 숫자, 점, 밑줄, 하이픈을 사용합니다. 같은 파일 이름의 가중치나 설정을 바꾸면 기존 결과와 섞지 않도록 중단합니다. 기존 결과를 보존하려면 가중치 파일 이름을 바꿔 새 모델 폴더에 저장합니다.

기본값은 VWorld 영상, 빛가람 z19 범위 `X 446760~446821, Y 207604~207658`의 3,410개 타일, `cuda:0`입니다. 영상은 메모리에서 읽으며, 옥상 추론에는 건물 외곽선과 3×3 타일 영상을 사용합니다. 결과는 `frontend/public/precomputed/panel/<가중치 이름>/`과 `frontend/public/precomputed/roof/<가중치 이름>/`에 확장자 없이 저장하고, 각 폴더의 `index.json`을 갱신합니다. 중단 후 같은 명령을 실행하면 성공한 타일을 건너뜁니다. 옥상 건물을 처리하지 못한 타일은 완료로 표시하지 않고 나중에 다시 시도합니다.

GPU 서버에서 생성한 결과를 적용하려면 **`precomputed/` 전체**를 배포용 프로젝트에 반영한 뒤 다시 배포해야 합니다. 모델 목록은 추론 환경의 결과 폴더들을 기준으로 생성되므로, 기존 모델도 유지하려면 다음 추론 전에 기존 `precomputed/` 전체를 GPU 환경에 복사합니다. 다른 구역은 `--bounds X_MIN Y_MIN X_MAX Y_MAX`로 z19 타일 범위를 지정하거나 `--tile-index <CSV>`로 좌표 목록을 지정할 수 있습니다. 다른 옵션은 `python -m pipelines.precompute --help`로 확인할 수 있습니다. 패널과 옥상을 따로 실행할 때는 `pipelines.precompute_panels`, `pipelines.precompute_roofs`를 사용합니다.

이번 전환에서 로컬 추론 결과는 생성하지 않았습니다. 실제 GPU 환경과 전체 영역 결과는 아직 검증하지 않았습니다. 원본 영상과 파생 결과의 이용·공개 조건은 따로 확인해야 합니다.

## 구조

- `frontend/`: 지도와 분석 화면.
- `pipelines/`, `modules/`: 영상 조회와 모델 학습·사전 추론.
- `models/`, `outputs/training/`: Git에서 제외하는 가중치와 학습 결과.
- `legacy/`: 이전 연구, 백엔드와 건축물대장 기능.
- `data/bitgaram/`: 공개에서 제외하는 기존 연구·학습 데이터.

## 현재 한계

옥상은 SAM 마스크와 건물 외곽선이 겹치는 부분으로 추정하며 평평한 지붕을 가정합니다. 옥탑·설비·나무·지붕 경사는 반영하지 않습니다. 주변 건물 그림자는 높이와 단순 형상으로 추정합니다. 높이가 없으면 15m, 층수만 있으면 층당 3.3m로 가정합니다. 건축물대장 기반 외곽선 검증은 적용하지 않습니다. VWorld 데이터 처리 허용 근거와 독립 지역 검증도 아직 확보되지 않았습니다. z19는 타일 줌 단계이며 실제 촬영 해상도와 촬영일을 보장하지 않습니다. 계산 결과는 시공 판단에 사용하면 안 됩니다.
