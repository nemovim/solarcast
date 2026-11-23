# Solar-panel inference

AI Hub 항공사진 폴리곤을 기존 검수 COCO와 합쳐 학습하고 빛가람에 추가 적응시키는 코드는
[AIHUB_TRAINING.md](AIHUB_TRAINING.md)를 참고하세요.

## 새 4인 마스킹 데이터로 파인튜닝

수합한 완료 이미지 804장용 진입점은 `bash finetune_masking.sh`입니다.
패널 있음·없음을 각각 8:2로 분할해 train 643장, validation 161장으로 학습합니다.
`--archives`로 네 ZIP에서 바로 시작하거나 `--prepare-only`로 GPU 없이 데이터·설정을 검증할 수 있습니다.
[실행 명령·데이터 분할·학습 설정 안내](MASKING_FINETUNE.md)를 참고하세요.

CUDA PyTorch가 설치된 서버에서 `3_panel_detection` 폴더로 이동해 실행한다. 각 스크립트는 필요한 패키지 설치, 모델 다운로드, 추론까지 수행한다.

```text
src/          추론, 다운로드, 파인튜닝 구현
models/       다운로드한 모델 파일
outputs/      새 추론/학습 결과 (archive/는 기존 결과)
```

```bash
# 전체 이미지
bash run_deeplabv3.sh
bash run_yolo.sh
bash run_nlr.sh

# 앞의 100장, confidence 0.4
bash run_nlr.sh --count 100 --confidence 0.4
```

세 추론 스크립트의 사용자 옵션은 동일하다.

- `--count N`: 정렬된 입력 중 앞의 N장만 처리한다. 생략하면 전체를 처리한다.
- `--confidence X`: 0~1 confidence cutoff. 기본값은 DeepLabV3 `0.50`, YOLO/NLR `0.25`이다.

입력은 `../data/bitgaram/raw_tiles/z19`로 고정되어 있다. 결과는 confidence별 `outputs/<model>/conf_<값>/`에 저장된다. NLR은 그 사이에 `original/` 또는 `finetuned_<실행시각>/`를 두어 모델별 결과를 구분한다. 중단 후 같은 명령을 실행하면 완료된 이미지는 건너뛴다.

```text
masks/            이미지별 합성 1-bit 마스크
instance_masks/   서로 분리된 instance별 1-bit 마스크
summary.csv       원본 파일명, XYZ/중심좌표, 크기, instance 수
instances.csv     instance별 confidence, bbox, 픽셀 수, 마스크 경로
```

탐지 결과를 지도용 GeoJSON으로 내보낸다. 타일마다 따로 추론하기 때문에 타일
경계에 걸친 패널이 여러 조각으로 쪼개져 나오는데(원본 결과 기준 2,257건 중 1,112건), 이
스크립트가 타일을 이어붙여 연결성분으로 병합한 뒤 WGS84 폴리곤으로 바꾼다.

```bash
python -m src.export_geojson                    # 기본 경로로 실행
python -m src.export_geojson --min-area 5.0     # 작은 성분을 더 걸러낸다
# 새 NLR 추론 결과를 사용하려면 입력 폴더를 명시한다.
python -m src.export_geojson --input outputs/nlr/original/conf_0p25 --output outputs/original_panels.geojson
```

기본 입력은 최종 AI-Hub 레이어를 생성한 원추론 폴더 `../data/solar_panel/final_neck_conf_0p20`이며, 팀이 별도로 보관한 summary.csv, instances.csv 및 마스크가 필요하다. 기본 출력은 `../outputs/3_panels/panels_final.geojson`이다.

파인튜닝은 검수 완료된 annotation을 COCO로 다시 내보낸 뒤 NLR 모델을 학습한다.

```bash
bash finetune_masking.sh                 # 최대 12 epochs, 자동 조기 종료
```

NLR 설치/학습에는 CUDA PyTorch와 `nvcc`가 필요하다.

v3는 backbone/neck을 고정하고 head를 학습한다. 검수된 빈 이미지도 학습에 포함하며,
전체 mask mAP 기준으로 best를 선택한다. FP32, LR 5e-5, patience 3으로 시작한다.
실행마다 `outputs/fine_tune/nlr_human_v3/<실행시각>/`에 데이터 snapshot과 결과가 저장된다.
`evaluation/report.json`에는 원본/v3의 빈 이미지 오탐률과 instance precision/recall이 나온다.
confidence는 validation에서만 선택한다. 이미 확인한 작은 test이므로 최종 검증은 새 지역이 필요하다.

추가 마스킹 없이 기존 데이터로 학습 방식 비교:

```bash
python -m src.compare_nlr --output outputs/fine_tune/nlr_compare/new_run
```

같은 데이터 snapshot, seed, LR, 최대 12 epochs와 patience 3으로 네 가지를 순서대로 비교한다.
`head_average` → `neck_average` → `neck_average_crop` → `neck_fresh_crop` 순서이며,
각 단계는 학습 범위, crop, 분류기 초기화 중 하나만 변경한다. Crop은 기존 이미지와
정답 마스크를 함께 자르는 학습 증강이며 검증·추론 이미지에는 적용하지 않는다.
자동 예측을 정답으로 추가하거나 검증·테스트 이미지를 학습에 넣지 않는다.

`outputs/fine_tune/nlr_compare/<실행시각>/comparison.json`에서 결과를 확인한다.
각 실험의 `train_console.log`에 출력과 오류가 저장된다. 다른 터미널에서
`tail -f <실험폴더>/train_console.log`로 진행을 확인할 수 있다.
모델 선택에는 validation만 사용하며, 선택된 모델만 test를 평가한다.
어느 모델도 검증 검출률 80%를 충족하지 못하면 `selected_experiment`는 `null`이고
`status`는 `recall_target_not_met`이다. 이 경우 test는 평가하지 않으며 성공으로 해석하지 않는다.
선택된 모델의 실험 폴더를 `NLR_RUN`에 지정하고 그 모델 report의 confidence로 추론한다.

중간 실패 후 같은 비교를 다시 실행할 때는 기존 snapshot으로 재시도한다:

```bash
RUN=outputs/fine_tune/nlr_compare/run-id  # 대상 비교 실행 폴더 이름으로 변경
python -m src.compare_nlr --dataset "$RUN/dataset/coco" --output "$RUN" --resume
```

완료된 실험은 재사용하며 실패한 실험은 처음부터 다시 학습한다. 실패 폴더는
`.failed_<시각>`으로 보관한다. 데이터·epochs·patience가 달라지면 재사용을 거부한다.
첫 검증에서 탐지가 0건이어도 bbox/segm mAP을 0으로 기록하고 학습을 이어간다.
계속 0이면 기존 patience에 따라 중단하며, 다른 평가 오류를 숨기지는 않는다.

의도적인 과적합 진단 (일반화 성능 개선을 보장하는 모드가 아님):

```bash
python -m src.train_nlr --overfit                 # 기본 100 epochs
python -m src.train_nlr --overfit --epochs 150    # 필요하면 더 길게
```

원본 NLR의 분류기 평균 초기화는 유지하고 backbone/neck/head 전체를 학습한다.
LR은 기존 5e-5(backbone은 0.1배)를 유지한다. Crop/색상/좌우반전 증강,
weight decay, early stopping을 끄고 지정한 epochs를 끝까지 실행한다.
학습에는 기존 train 107장만 사용하며 빈 이미지도 유지한다. val/test를 train에 섞지 않는다.
여러 조건을 동시에 바꾸는 진단이므로 기존 비교 실험과 달리 개별 요인의 효과를 분리할 수 없다.

`outputs/fine_tune/nlr_overfit/<실행시각>/`에 별도로 저장한다.
validation-best checkpoint도 보관하지만 진단·추론용 `final_model.pth`는 마지막 epoch 모델이다.
`train_console.log`, `evaluation_console.log`, `training_setup.json`에 설정과 진행을 남긴다.
`evaluation/report.json`의 `finetuned.train_diagnostic`과 `finetuned.selected_val`을 비교한다.
train 지표는 학습 데이터 암기 여부를 확인하는 용도이며 confidence는 val에서만 선택한다.
test는 자동 평가하지 않고 전체 지역 데이터 추출도 실행하지 않는다.
train은 높고 val은 낮으면 과적합 징후이며, train도 낮으면 학습 설정/정답/평가 정합성을 추가 점검한다.
일반 실행의 `best_model.pth`와 혼동하지 말고 이 모드는 `final_model.pth`를 사용해야 한다.

학습된 모델 실행 (학습 완료 시 출력된 폴더와 report의 confidence 사용):

```bash
NLR_RUN=outputs/fine_tune/nlr_human_v3/<실행시각> bash run_nlr.sh --confidence 0.4
```

횡단보도·창문 오탐은 사람이 태양광 없음으로 확인한 타일을 빈 GT로 추가해야 한다.
태양광이 함께 있는 타일에서는 실제 태양광만 마스킹한다. 모델 예측만 보고 빈 GT를 만들지 않는다.
