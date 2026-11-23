# 2. 옥상 면적 및 유휴부지 탐색 모듈 (Roof Detection & Layout)

건물 Footprint(지상 외곽선)의 시차 오차를 보정하고, SAM(Segment Anything Model)을 활용하여 실제 옥상 설치 가능 영역(유휴부지)을 정밀 세그멘테이션한 뒤 400W 규격 모듈을 자동 배치하는 모듈입니다.

---

### 주요 구성 및 역할

1. **`roof.py`**:
   * SAM ViT-H 기반 건물별 옥상 정밀 분할
   * 지표면 footprint 기반 힌트 좌표를 활용하여 옥상 경계선 마스크 추출
2. **`available.py`**:
   * 옥상 내 구조물(엘리베이터실, 계단탑 등) 및 음영 유발 요소를 제외한 순수 평탄 설치 가능 영역(A_roof) 판정
3. **`export.py`**:
   * 옥상 설치 가능 영역 인계 데이터 추출 (`buildings.geojson`)
4. **`build_available_geojson.py`**:
   * SAM 마스크 이미지를 WGS84 좌표계의 벡터 폴리곤 GeoJSON으로 변환 (`available.geojson`)
5. **`build_panel_layout.py`**:
   * 설치 가능 영역 내에 1.0m × 2.0m (400W) 규격 태양광 패널 모듈을 줄 간격 1m로 자동 격자 배치 (`panel_layout.geojson`)

---

### 입출력 데이터

* **입력**: `data/bitgaram/vector/buildings.geojson` (건물 외곽선), `data/bitgaram/raw_tiles/z19/` (항공 타일)
* **산출물**:
  * `data/rooftop_area/sam_vith_v1/buildings.geojson`
  * `data/rooftop_area/web/available.geojson` (웹 지도용 옥상 유휴부지 폴리곤)
  * `data/rooftop_area/web/panel_layout.geojson` (웹 지도용 400W 모듈 배치 폴리곤)

---

### 실행 예시

```bash
# 옥상 유휴부지 폴리곤 GeoJSON 생성
python 2_roof_detection/build_available_geojson.py

# 유휴부지 내 400W 태양광 모듈 자동 배치 GeoJSON 생성
python 2_roof_detection/build_panel_layout.py
```
