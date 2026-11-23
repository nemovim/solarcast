# 1. 항공 사진 & 지도 데이터 모듈 (Aerial Imagery & Geocoding)

VWorld WMTS 고해상도(z19) 항공영상 수집, 좌표계 변환, 축소 피라미드 타일 생성 및 건축물대장 속성 결합을 담당하는 모듈입니다.

---

### 주요 구성 및 역할

1. **`tiles.py`**:
   * WMTS 타일 좌표 $(x, y, z=19)$ ↔ WGS84 경위도(EPSG:4326) / Web Mercator(EPSG:3857) 상호 변환 핵심 라이브러리
2. **`where.py`**:
   * 타일 번호, 경위도 좌표, 512×512 칩 ID 간의 위치 상호 조회 도구
   * VWorld, 카카오맵, 구글맵 위성 대조 URL 생성
3. **`ledger.py`**:
   * 전남광주통합특별시 신코드(12170) 기반 건축물대장 공공데이터 API 연동
   * 건물 footprint 면적과 대장 건축면적을 최적 할당(`linear_sum_assignment`)하여 동별 층수 및 높이(`heit`) 속성 조인
4. **`build_tile_pyramid.py`**:
   * 원본 z19 타일(3,410장)로부터 웹 지도 줌아웃용 z14~z18 축소 타일 생성 (`.webp`)

---

### 실행 예시

```bash
# 특정 타일의 경위도 및 지도 대조 링크 확인
python 1_aerial_imagery/where.py --tile 446788 207627

# 특정 경위도가 속한 타일 조회
python 1_aerial_imagery/where.py --lonlat 126.7853 35.0235

# 웹 시연용 축소 타일 피라미드 생성 (Pillow 필요)
python 1_aerial_imagery/build_tile_pyramid.py
```
