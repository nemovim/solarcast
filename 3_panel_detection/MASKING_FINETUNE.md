# 새 마스킹 데이터로 NLR 파인튜닝

기존 `src/train_nlr.py`의 가중치 이식·학습·best 모델 선택 기능을 재사용합니다.
새 진입점은 `finetune_masking.sh` / `src.finetune_masking`이며, 이전 `human_v1` 기록은 읽지 않습니다.

## 사용하는 데이터와 분할

기본 입력은 `src/export_masking.py`에 지정된 `annotations_merged` 아래의 병합 데이터 배치입니다.
이 중 **done(패널 있음)**과 **empty(패널 없음)**만 사용합니다.
보류 5장, 작업 중 1장, 미작업 2,600장은 제외합니다.

패널 있음과 패널 없음을 각각 섞고, 각 그룹에서 80%를 train으로, 나머지를 validation으로
배정합니다. 기본 seed는 `2026`이며 같은 입력·seed에서는 같은 분할을 만듭니다.
이미지 단위 분할이므로 한 이미지 안의 영역이 train과 validation으로 갈라지지 않습니다.

| 구분 | 전체 | train | validation |
| --- | ---: | ---: | ---: |
| 패널 있음 | 124 | 99 | 25 |
| 패널 없음 | 680 | 544 | 136 |
| 합계 | 804 | 643 | 161 |

`train=floor(n×0.8)`, `validation=나머지`로 정수 개수를 정합니다.
이 데이터에는 패널 영역이 262개 있으며 기본 seed에서 train 219개·validation 43개입니다.
별도 test 분할은 만들지 않으며 validation을 test로 복제하지 않습니다.

## GPU 서버에서 실행

저장소 최상위 폴더에서:

```bash
cd 3_panel_detection
bash finetune_masking.sh
```

기존 파인튜닝과 같은 CUDA PyTorch 환경이 필요합니다. `setup_nlr_env.sh`는 기존 스크립트에서
분리한 동일한 의존성 설치 절차를 실행합니다. MMCV 연산을 설치해야 할 때는 `nvcc`가 필요합니다.
기본 NLR 설정·원본 체크포인트가 없으면 다운로드합니다.

기존 804장 학습의 정답 마스크와 학습된 가중치는 이 배포본에 포함되어 있지 않습니다. 팀의 기존 수합 폴더를 `--merged`로 지정하거나 네 파트 ZIP을 사용합니다. 아래 예시는 ZIP을 `3_panel_detection/masking/`에 둔 경우입니다.

```bash
cd 3_panel_detection
bash finetune_masking.sh --archives masking/part-1.zip masking/part-2.zip masking/part-3.zip masking/part-4.zip
```

이 명령은 ZIP 검증·수합 → 완료 데이터 변환 → 설정 검증 → 학습 → validation 평가를 실행합니다.
서버에도 `masking/` 소스와 `assignments.json`, 원본 JPEG 3,410장, 네 ZIP이 있어야 합니다.
ZIP과 수합 결과는 자동으로 원격 서버에 전송되지 않습니다.

이미 학습 환경이 준비되어 설치 단계를 건너뛰려면:

```bash
python -u -m src.finetune_masking --archives masking/part-1.zip masking/part-2.zip masking/part-3.zip masking/part-4.zip
```

## 데이터와 설정만 미리 검증하기

NumPy·Pillow가 있는 Python 환경에서 GPU 없이 실행할 수 있습니다.
이 모드에서는 모델 다운로드, CUDA 패키지 설치, 실제 학습을 하지 않습니다.

```bash
cd 3_panel_detection
python -u -m src.finetune_masking --prepare-only
```

ZIP에서 시작할 때도 `--prepare-only`를 함께 지정할 수 있습니다.
CPU 검증에서 생성한 폴더와 다른 새 출력 폴더로 실제 학습을 실행합니다.
같은 입력과 seed를 쓰면 train/validation 분할은 동일합니다.

## 학습 설정

| 설정 | 기본값 |
| --- | --- |
| 모델 | 기존 NLR RTMDet-Ins-X, 태양광 단일 클래스 |
| 초기 가중치 | 원본 5클래스 NLR, 분류 채널 평균으로 단일 클래스 초기화 |
| 학습 범위 | neck + head, backbone 고정 (`--feature-mode neck`) |
| optimizer | AdamW, LR `5e-5`, weight decay `0.05`, FP32 |
| batch / gradient 누적 | 1장 / 4회 |
| 최대 epochs / 조기 종료 patience | 12 / 3 |
| 모델 선택 | validation `coco/segm_mAP` 최고 체크포인트 |
| 증강 | 기존 640px resize, 색상 변형, 좌우 반전 |
| 빈 이미지 | 학습·검증 모두 유지, `filter_empty_gt=False` |

기존 비교에서 사용했던 neck 학습 모드를 기본으로 하며, 이전의 과적합 진단 모델에서 이어
학습하지 않습니다. 다른 범위는 `--feature-mode head` 또는 `--feature-mode full`로 지정합니다.

```bash
bash finetune_masking.sh --epochs 30 --patience 5 --feature-mode neck
```

`--seed`, `--learning-rate`, `--batch-size`, `--accumulate`, `--workers`, `--device`도 지정할 수 있습니다.
입력 경로는 `--merged` 또는 `--archives`, 원본 경로는 `--images`로 지정합니다.
`--checkpoint`를 바꿀 경우 원본처럼 5클래스 분류 채널을 가진 동일 구조 가중치를 사용해야 합니다.

## 마스크와 검증

완료된 PNG가 저장 좌표와 같은지 먼저 검사합니다. 직접 그린 다각형 하나를 instance 하나로
유지하고, 작은 영역을 임의 면적 기준으로 제거하지 않습니다. 픽셀 중심 기준으로 생성한
마스크를 COCO uncompressed RLE로 저장하고 `poly2mask=True`로 bitmap GT를 불러옵니다.
이 방식은 [MMDetection 3.3.0의 마스크 로더](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/datasets/transforms/loading.py)에 맞춥니다.

패널 없음 이미지는 COCO `images`에 포함하되 해당 이미지의 `annotations`는 비워 둡니다.
원본/마스크 크기, 좌표와 PNG, train/validation ID 중복, 서로 다른 ID의 동일 이미지 내용 중복을 검사합니다.
validation에서 confidence를 선택하고, 원본 모델과 학습 모델의 recall·precision 및 빈 이미지 오탐을 비교합니다.
validation은 모델·threshold 선택에 사용되므로 독립 test 성능으로 해석하지 않습니다.

## 결과 위치

기본 경로는 `3_panel_detection/outputs/fine_tune/nlr_masking_v2/<실행시각>/`입니다.
기존 결과를 덮어쓰지 않으며 `--output`으로 새 경로를 지정할 수 있습니다.

```text
<실행시각>/
├─ source/                    # --archives 사용 시 수합한 원본 기록
├─ dataset/
│  ├─ coco/train/             # images/, masks/, annotations.json
│  ├─ coco/val/
│  ├─ reviewed_annotations/   # 완료 기록 snapshot
│  ├─ split_manifest.csv      # 이미지별 분할·상태·SHA-256
│  └─ split_report.json       # 클래스별 개수·제외 상태·검증 결과
├─ run_plan.json              # 실행 명령과 데이터 fingerprint
├─ nlr_config.py
├─ train_console.log
├─ training_setup.json
├─ best_model.pth
├─ inference_config.py
└─ evaluation/
   ├─ report.json
   └─ per_image.csv
```

`--prepare-only`에서는 dataset·설정·실행 계획만 생성되며 `best_model.pth`는 아직 없습니다.
실제 학습 후 `best_model.pth`와 `inference_config.py`를 함께 사용합니다.
이전 `final_model.pth`(과적합 진단)와 구분합니다.

## 학습 후 평가만 재실행하기

학습 후 `evaluate_nlr.py`의 `init_detector`에서 dataset `type`이 없다는 오류가 났다면,
평가·추론 코드 수정본을 서버에 반영하고 아래처럼 평가만 다시 실행합니다.
저장된 `best_model.pth`를 사용하므로 재학습이나 데이터 재분할은 하지 않습니다.
수정본은 이전에 생성한 pipeline-only `inference_config.py`도 지원합니다.

```bash
cd 3_panel_detection
RUN_ID=run-id  # 대상 실행 폴더 이름으로 변경
RUN=outputs/fine_tune/nlr_masking_v2/$RUN_ID
python -u -m src.evaluate_nlr --run "$RUN" --dataset "$RUN/dataset/coco" --val-only --device cuda:0 --hard-exit-after-success
```

자신의 실행 폴더가 다르면 `RUN`을 바꿉니다. 결과는 해당 폴더의 `evaluation/report.json`과
`evaluation/per_image.csv`에 저장됩니다. 이 명령은 평가만 실행하므로 `run_plan.json`의
기존 실패 상태는 자동으로 변경하지 않습니다.

## 원본·neck·full 결과 비교

이미 생성한 평가 보고서를 합쳐 비교합니다. GPU나 재학습은 필요하지 않습니다.
각 실행 폴더의 validation 원본 파일 내용과 정답을 비교하고, 서로 다르면 중단합니다.
모델별로 선택된 confidence와 공통 confidence(기본 0.15) 두 조건을 따로 출력합니다.

```bash
cd 3_panel_detection
NECK_RUN_ID=neck-run-id  # 각 실행 폴더 이름으로 변경
FULL_RUN_ID=full-run-id
NECK=outputs/fine_tune/nlr_masking_v2/$NECK_RUN_ID
FULL=outputs/fine_tune/nlr_masking_v2/$FULL_RUN_ID
python -m src.compare_nlr_results --run neck "$NECK" --run full "$FULL" --output outputs/comparisons/neck_vs_full
```

출력은 `comparison.md`, `comparison.json`, `selected_threshold.csv`, `common_threshold.csv`입니다.
원본 모델은 첫 보고서에서 한 번만 가져옵니다. 보고서 간 원본 평가 결과가 다르면 경고를 남깁니다.
출력 폴더가 이미 있으면 새 이름을 지정합니다. `--common-confidence 0.25`로 공통 기준을 바꿀 수 있습니다.

다른 모델도 `--run 이름 실행폴더`로 추가할 수 있지만, 해당 폴더에 같은 validation의 평가 결과가
있어야 합니다. 예전 human_v1 모델처럼 검증 데이터가 다른 경우에는 먼저 같은 데이터로 재평가해야 합니다.
모든 표는 모델과 confidence 선택에 사용한 validation 결과이며 독립 test 성능은 아닙니다.

## CPU 회귀 테스트

```bash
cd 3_panel_detection
python -m unittest discover -s tests -p test_masking_finetune.py -v
python -m unittest discover -s tests -p test_compare_nlr_results.py -v
```

## 이전 파인튜닝 모델까지 같은 데이터로 비교

서버에서 예전 모델 후보를 먼저 확인합니다. 현재 Windows 작업 폴더에는 해당 체크포인트가 없습니다.

```bash
cd 3_panel_detection
find outputs/fine_tune -type f \( -name inference_config.py -o -name best_model.pth -o -name final_model.pth \) | sort
```

아래 `OLD`는 위 목록에서 확인한 예전 실행 폴더로 지정합니다. 과적합 진단 실행은 `final_model.pth`, 일반 실행은 `best_model.pth`를 사용합니다. 설정 파일과 체크포인트는 반드시 같은 실행에서 가져옵니다.

```bash
FULL_RUN_ID=full-run-id  # 각 실행 폴더 이름으로 변경
NECK_RUN_ID=neck-run-id
FULL=outputs/fine_tune/nlr_masking_v2/$FULL_RUN_ID
NECK=outputs/fine_tune/nlr_masking_v2/$NECK_RUN_ID
OLD=outputs/fine_tune/nlr_overfit/실제_이전_실행폴더
OLD_CHECKPOINT="$OLD/final_model.pth"

python -u -m src.evaluate_nlr \
  --run "$FULL" --dataset "$FULL/dataset/coco" --val-only \
  --extra-model neck "$NECK/inference_config.py" "$NECK/best_model.pth" \
  --extra-model previous "$OLD/inference_config.py" "$OLD_CHECKPOINT" \
  --output "outputs/comparisons/four_models_$(date +%Y%m%d_%H%M%S)" \
  --device cuda:0 --hard-exit-after-success
```

네 모델을 **이번 full 실행의 동일한 validation 이미지·정답**으로 순차 평가하며 재학습하지 않습니다. `original`은 기본 모델, `finetuned`는 이번 full, `neck`은 이번 neck, `previous`는 이전 모델입니다. 기존 실행의 평가 결과는 덮어쓰지 않습니다.

결과 폴더의 `comparison.md`에서 비교표를 확인합니다. `selected_threshold.csv`는 모델별로 validation에서 선택한 confidence, `common_threshold.csv`는 모두 confidence 0.15를 적용한 결과입니다. `report.json`에는 전체 confidence 곡선과 사용한 설정·모델 경로가 저장됩니다. 이 비교는 mask IoU 0.5 기준 TP·FP·FN, precision·recall·빈 이미지 오탐 지표이며 COCO mAP를 새로 계산하지는 않습니다. 이전 모델의 학습에 해당 이미지가 포함됐다면 독립적인 일반화 성능으로 해석할 수 없습니다.

## 이번 데이터로 full 과적합 실험

기본 NLR 가중치에서 새로 full 학습합니다. 기존 full 체크포인트를 이어 학습하는 기능은 아닙니다. 동일한 원본 ZIP·seed를 사용하면 기존과 동일하게 패널 유무별 8:2 분할을 재현합니다.

```bash
cd 3_panel_detection
bash finetune_masking.sh --overfit \
  --archives masking/part-1.zip masking/part-2.zip masking/part-3.zip masking/part-4.zip
```

`--overfit`은 full, 기본 100epoch, 증강 없음, weight decay 0, 조기 종료 없음으로 설정합니다. 학습률은 기본 5e-5입니다. 다른 길이는 `--epochs 50`처럼 지정합니다. 검증 데이터는 학습에 넣지 않으며, 패널 없는 학습 이미지도 유지합니다.

결과는 별도 `outputs/fine_tune/nlr_masking_v2_overfit/<실행시각>/`에 저장합니다. 마지막 epoch의 `final_model.pth`를 자동 평가하며, validation 기준 비교 결과와 별도 train 진단을 생성합니다. confidence는 validation에서만 선택합니다. 기존 일반 full의 `best_model.pth`와 혼동하지 마세요.

과적합 모드는 체크포인트를 10epoch마다 저장하고 모두 유지합니다(`epoch_10.pth`, `epoch_20.pth`, …). 마지막 epoch도 항상 저장하므로 `--epochs 25`라면 `epoch_25.pth`도 생성됩니다. validation은 매 epoch 수행하지만 별도의 validation-best 모델은 저장하지 않습니다. 마지막 모델은 `final_model.pth`로도 복사해 자동 평가합니다. 이 변경은 코드를 반영한 뒤 새로 시작하는 실행에 적용됩니다.
