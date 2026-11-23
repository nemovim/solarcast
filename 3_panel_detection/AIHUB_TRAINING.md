# AI Hub 항공사진으로 NLR 추가 학습

TIFF는 축소 미리보기를 제외한 원본 페이지가 정확히 한 개일 때만 사용합니다.
원본이 여러 페이지이거나 원본을 특정할 수 없으면 파일 전체를 건너뛰고 계속합니다.
원본 이미지·라벨 파일은 삭제하지 않으며 해당 파일의 조각을 생성하지 않습니다.
제외 파일은 `import_report.json`의 `skipped_ambiguous_tiffs`에 기록합니다.
`skipped_ambiguous_tiff_count`는 제외한 원본 TIFF 수입니다.

중복 정책은 **먼저 나온 이미지와 라벨 유지, 나중 중복 제거**입니다.
픽셀이 완전히 같으면 마스크 차이와 관계없이 뒤의 조각을 버립니다.
마스크 합치기와 충돌 제외 정책은 제거했습니다.
폴더 모드는 파일명 정렬 순서, 각 원본은 위에서 아래·왼쪽에서 오른쪽 순서로 처리합니다.
중복 해시는 저장된 PNG 픽셀과 재확인합니다.
`import_report.json`의 `duplicate_tiles_skipped`, `duplicate_tiles`에
유지한 이미지, 버린 출처·좌표, 라벨 차이 여부를 기록합니다.

## 권장: 변환과 학습 분리

변환은 빛가람 파일 없이 AI Hub만 처리합니다. `solar-panel`에서:

```bash
python -u -m src.prepare_aihub --aihub-root ../data/aihub --output ../data/aihub/coco_256
```

출력은 `coco_256/train/images/*.png`, `coco_256/train/annotations.json`,
`import_report.json`, `source_manifest.csv`입니다. 512px를 네 조각으로 나누며
빈 타일도 포함합니다. `valid_*`는 현재 변환에서 사용하지 않습니다.
빛가람 파일을 읽거나 복사하지 않습니다. 이미 존재하는 출력은 덮어쓰지 않습니다.

나중에 학습할 때 두 경로를 별도로 지정합니다:

```bash
VAL_RUN_ID=run-id  # 검증 데이터로 사용할 NLR 실행 폴더 이름
python -u -m src.train_nlr \
  --dataset ../data/aihub/coco_256 \
  --val-dataset "outputs/fine_tune/nlr_masking_v2/$VAL_RUN_ID/dataset/coco" \
  --output outputs/aihub_train/run01 \
  --bitmap-masks --feature-mode neck --epochs 30 --patience 5
```

`--dataset/train`만 학습하고 `--val-dataset/val`만 검증합니다.
별도 test는 평가하지 않습니다. CUDA 환경과 원본 NLR 파일은 기존처럼 준비해야 합니다.
학습 시작 전 두 데이터의 동일 이미지 픽셀 중복도 검사합니다.
아래 병합 방식은 이전 실행 호환용이며 이 권장 흐름에서는 필요하지 않습니다.

## 압축 해제한 폴더에서 바로 변환

`data/aihub/` 아래 `train_data/`, `train_label/`, `train_meta/`,
`valid_data/`, `valid_label/`, `valid_meta/` 구조이면 CSV 없이 실행할 수 있습니다.
저장소 루트에서:

```bash
cd solar-panel
# 빛가람 COCO가 아직 없을 때만 실행
bash finetune_masking.sh --prepare-only --output outputs/local_prepared

python -u -m src.prepare_aihub \
  --aihub-root ../data/aihub \
  --local-dataset outputs/local_prepared/dataset/coco \
  --train-source aihub \
  --output outputs/aihub_only/coco
```

`train_data`의 TIF와 `train_label`의 JSON을 확장자 제외 파일명으로 연결합니다.
하위 폴더도 검색하며 누락·중복 파일명은 오류로 처리합니다.
AI Hub `valid_*`와 메타데이터는 사용하지 않습니다. validation은 빛가람 val입니다.
512px 이미지는 256px 네 조각으로 변환하며, 진행률은 원본 100장마다 출력합니다.
이 CLI의 기본 학습 소스는 `aihub`이며, 혼합 학습 데이터가 필요할 때는
`--train-source mixed`와 별도 출력 경로를 명시하세요.

## AI Hub만 학습하고 빛가람으로 검증

현재 validation은 항상 직접 마스킹한 빛가람 COCO의 val만 사용합니다.
manifest의 외부 val 행은 제외하며 학습으로 재배정하지 않습니다.
다음 옵션은 빛가람 train도 제외하고 AI Hub train의 256px 조각만 학습합니다.

```bash
python -m src.prepare_aihub \
  --manifest ../data/aihub/manifest.csv \
  --local-dataset outputs/local_prepared/dataset/coco \
  --train-source aihub --output outputs/aihub_only/coco

python -u -m src.train_nlr --dataset outputs/aihub_only/coco \
  --output outputs/aihub_only_run --train-val-only --bitmap-masks \
  --feature-mode neck --epochs 30 --patience 5

python -u -m src.evaluate_nlr --dataset outputs/local_prepared/dataset/coco \
  --run outputs/aihub_only_run --val-only
```

빛가람 val은 모델·confidence 선택에 반복 사용되므로 독립 test 성능은 아닙니다.
이후 현지 추가 학습에는 빛가람 train만 쓰고 같은 val을 계속 유지합니다.
기존 데이터셋은 자동 변경되지 않으므로 새 출력 경로로 다시 변환하세요.

## 세 실험 자동 비교

아래 데이터 변환을 완료한 후 한 명령으로 세 실험을 순서대로 실행할 수 있습니다.
모델 파일은 `python -m src.download nlr`로 먼저 준비하세요.

```bash
python -u -m src.compare_aihub \
  --mixed-dataset outputs/aihub_mixed/coco \
  --local-dataset outputs/local_prepared/dataset/coco \
  --output outputs/aihub_compare/run01
```

1. `mixed_only/`: 원본 NLR → 외부 + 빛가람 혼합 학습
2. `local_only/`: 원본 NLR → 빛가람만 학습
3. `mixed_then_local/`: 1번 best 모델 → 빛가람 추가 학습

세 결과를 모두 **동일한 빛가람 validation**에서 평가합니다.
각 폴더의 `best_model.pth`, `evaluation/report.json`, `evaluation/per_image.csv`를
따로 보관합니다. 최상위 `comparison.csv`는 모델별 선택 threshold의 비교,
`comparison_conf_0p15.csv`는 동일 threshold 0.15 비교입니다.
`comparison.json`에는 세 학습 모델과 원본 NLR 기준선까지 담습니다.
각 단계는 순차 실행되며 앞 단계 실패 시 중단합니다. `run_plan.json`과 단계별 log에
상태를 남기고 기존 출력 폴더는 덮어쓰지 않습니다. 자동 재개는 지원하지 않습니다.

`--prepare-only`를 추가하면 데이터 검사와 실행 명령 저장까지만 수행합니다.
이 모드의 출력도 새 경로로 지정하고 실제 실행에는 별도 출력 경로를 쓰세요.
기본 혼합/현지만 학습은 각각 최대 30 epoch, 추가 학습은 최대 20 epoch입니다.
둘 다 한 모델은 학습량도 더 많으므로 데이터 효과만 분리하는 비교는 아닙니다.
실제 실행 전 `setup_nlr_env.sh`로 CUDA 환경을 준비해야 합니다.

기존 CUDA/MMDetection 환경을 사용합니다. 저장소 루트에서 `cd solar-panel` 후 실행합니다.
데이터는 AI Hub 데이터셋 71348에서 사용자가 신청·다운로드하고 ZIP을 해제해야 합니다.
이 코드는 다운로드나 실제 GPU 학습을 자동 시작하지 않습니다.

## 1. 빛가람 COCO 준비

기존에 생성한 `dataset/coco`가 있으면 그대로 사용합니다. 없으면:

```bash
bash finetune_masking.sh --prepare-only --output outputs/local_prepared
```

아래 예시의 로컬 경로는 `outputs/local_prepared/dataset/coco`입니다.
이 데이터의 기존 train/val 배정을 유지합니다. 인접한 빛가람 타일의 공간 누수는
별도로 점검해야 하며, 최종 성능 평가에는 학습/검증과 분리된 구역을 사용하세요.

## 2. 외부 데이터 manifest 작성

`../data/aihub/manifest.csv` 예시(경로는 CSV 위치 기준, 절대경로도 가능):

```csv
image,label,split,group
images/site_a_001.tif,labels/site_a_001.json,train,site_a
images/site_a_002.tif,labels/site_a_002.json,train,site_a
images/site_b_001.tif,labels/site_b_001.json,val,site_b
```

`group`은 같은 발전소·인접 촬영 구역을 묶는 이름입니다. 같은 구역의 12cm/25cm
영상도 같은 group과 split에 배정하세요. 공식 분할을 사용할 때도 구역 중복을 확인하세요.
자동으로 파일명에서 구역을 추측하지 않습니다. 데이터 전체 대신 선택한 수천 장만
manifest에 적어서 소규모 비교를 시작할 수 있습니다.

25cm/12cm **발전현황 항공·드론**의 `Solar panel` 폴리곤만 사용합니다.
SAR box, 적지분석 Roof/Top/Structure, 빈 외부 라벨은 오류로 처리합니다.
라벨은 공식 `Learning Data Info.objects`의 `annotation_type=polygon`,
`class_name=Solar panel`, `points=[x,y,...]` 구조를 기대합니다.
실제 다운로드 파일이 다른 구조라면 샘플을 확인한 후 파서를 수정해야 합니다.

```bash
python -m src.prepare_aihub \
  --manifest ../data/aihub/manifest.csv \
  --local-dataset outputs/local_prepared/dataset/coco \
  --output outputs/aihub_mixed/coco
```

기본값은 256×256 타일입니다. 25cm급 512×512 이미지를 확대·축소 없이 네 조각으로
나누고, 폴리곤을 원본 크기로 래스터화한 뒤 마스크도 정확히 같은 위치로 자릅니다.
경계에 걸친 패널은 각 조각에서 보이는 영역만 남기고 bbox/면적을 다시 계산합니다.
패널 없는 조각도 빈 GT로 유지합니다. 원본의 네 조각은 같은 train/val과 group을
유지합니다. `import_report.json`의 crop_box에 원본 기준 좌표를 기록합니다.
12cm급 1024×1024는 16조각이 되지만 지상 해상도는 여전히 12cm이므로, 현재
25cm급과의 크기 정합 실험에서는 25cm급만 사용하세요.
TIF를 무손실 RGB PNG로 저장하고 폴리곤을 기존 마스킹과 같은 COCO RLE로 변환합니다.
출력에는 이미지·라벨 출처, 픽셀 SHA-256, 그룹, 통계를 기록합니다.
이미지 픽셀 중복은 먼저 나온 항목을 유지하고 이후 항목을 제거합니다. 다중밴드/16비트 영상은
자동 정규화하지 않으므로 명시적으로 RGB 변환한 후 사용하세요.
기존 출력은 덮어쓰지 않습니다. 실패한 부분 출력도 재사용하지 말고 새 경로를 지정하세요.

## 3. 외부 + 빛가람 혼합 학습

```bash
# GPU 없이 COCO 통계와 학습 설정 생성 검증
python -m src.train_nlr --dataset outputs/aihub_mixed/coco \
  --output outputs/aihub_preflight --train-val-only --bitmap-masks --validate-only

# 기존 환경이 없을 때만 설치
bash setup_nlr_env.sh
python -m src.download nlr

python -u -m src.train_nlr --dataset outputs/aihub_mixed/coco \
  --output outputs/aihub_stage1 --train-val-only --bitmap-masks \
  --feature-mode neck --epochs 30 --patience 5
```

기존 NLR 5클래스 체크포인트를 평균 초기화해 패널 한 클래스로 학습합니다.
빛가람의 검수된 음성 이미지를 함께 포함합니다. 외부 데이터가 많으면 현지 이미지가
상대적으로 적게 노출되므로, 다음 현지 추가 학습과 현지 validation을 함께 비교하세요.

## 4. 빛가람 데이터로 이어 학습

```bash
python -u -m src.train_nlr \
  --dataset outputs/local_prepared/dataset/coco \
  --checkpoint outputs/aihub_stage1/best_model.pth --classifier-init single \
  --output outputs/aihub_stage2 --train-val-only --bitmap-masks \
  --feature-mode neck --learning-rate 1e-5 --epochs 20 --patience 5

python -u -m src.evaluate_nlr \
  --dataset outputs/local_prepared/dataset/coco \
  --run outputs/aihub_stage2 --val-only
```

`single`은 1클래스 모델을 그대로 로드하며 일부 가중치가 누락되면 중단합니다.
optimizer/scheduler를 재개하는 resume가 아니라 새 학습 단계입니다.
best는 validation mask mAP으로 선택하며 confidence도 validation에서 선택합니다.
학습 전후 비교는 기존 빛가람 모델과 동일한 validation에서 수행하세요.
정확도 개선은 실제 GPU 실험과 별도 구역 평가로 확인해야 합니다.
