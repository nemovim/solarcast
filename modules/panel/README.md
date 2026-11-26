# 패널 모델

- `training/`: NLR RTMDet-Ins 학습과 데이터 누출 점검.
- `inference/`: RGB 이미지 한 장에 대한 패널 마스크·신뢰도·상자 추론.
- `contracts.py`: 단일 이미지 추론 결과의 자료형.
- `configs/`: 학습·추론에 사용하는 모델 설정. 가중치는 루트 `models/panel/`에 별도로 둔다.

영상 조회와 모델을 연결하는 GPU 사전 추론 명령은 루트의 `pipelines/precompute_panels.py`에 있다. 실행 방법과 산출물 반영 절차는 루트 README를 따른다. 사전 추론은 팀이 지정한 NLR 설정과 가중치가 필요하고, CUDA PyTorch·MMCV·MMDetection 호환 환경을 별도로 준비해야 한다. 타일 경계의 탐지 품질과 결과의 독립 지역 정확도는 검증되지 않았다.
