# PCQM4Mv2 데이터셋 생성 파이프라인

이 문서는 PCQM4Mv2 데이터셋 (`/workspace/DATASET/PCQM4M_V2/aug5`) 생성을 위한 전체 파이프라인 과정을 설명합니다.  
총 3단계로 구성되어 있으며, 각 단계의 목적과 관련 스크립트를 정리하였습니다.

---

## 1단계: `multi_conf`를 이용한 효율적인 좌표 생성

- 각 분자에 대해 다수의 conformer를 생성하기 위해 `multi_conf`를 사용합니다.
- 좌표의 다양성과 현실성을 높이는 것이 목적입니다.
- 관련된 변경사항:
    - `coord_utils.py`
    - `data_utils.py`
    - `pcqm4mv2_lmdb.py` (1, 2번 섹션)

---

## 2단계: 증강(augmentation) 및 처리 방식 개편

### 배경

처음에는 각 분자마다 10개의 conformer를 생성하려 했으나, 처리 속도가 너무 느려 중단하게 되었습니다. 이에 따라 다음과 같은 흐름으로 데이터를 생성 및 병합하게 되었습니다:

1. 분자당 10개 conformer 생성 시도 (속도 문제로 중단됨)
2. `shrink_lmdb.py`를 이용해 10개 중 5개만 사용하고, 처리 완료된 분자 표시
3. `find_unprocessed.py`로 처리되지 않은 분자만 골라 새로운 SDF 파일 생성
4. 해당 SDF에 대해 다시 `multi_conf`를 사용해 5개 conformer 생성
5. 2번과 4번 결과를 `merge_shrink_remain.py`로 병합 (기본 경로도 수정됨)
6. 안정성을 위한 추가 수정:
   - key가 정수가 아닌 경우가 있어, 키 유효성 체크 코드 추가
   - edge feature가 단방향으로만 구성되어 있어, interleave 처리 적용

### 관련 스크립트

- `shrink_lmdb.py`
- `find_unprocessed.py`
- `merge_shrink_remain.py`
- `utils.py` (3-1번)
- `pcqm4mv2_lmdb.py` (6–9-1번 섹션)

---

## 3단계: `tqdm`을 통한 진행 상황 시각화 개선

- 데이터셋 생성 과정에서의 진행 상황을 보기 쉽도록 `tqdm` 사용 방식을 개선했습니다.
- 출력 포맷 통일 및 가독성 향상이 목적입니다.
- 관련 변경사항:
  - `pcqm4mv2_lmdb.py` (3–5번, 9-2번, 10번 섹션)

---

## 요약

| 단계 | 주요 작업                     | 관련 스크립트 및 파일 |
|------|------------------------------|------------------------|
| 1단계 | 좌표 생성 (`multi_conf`)       | `coord_utils.py`, `data_utils.py`, `pcqm4mv2_lmdb.py` (1-2) |
| 2단계 | 증강 및 병합                  | `shrink_lmdb.py`, `find_unprocessed.py`, `merge_shrink_remain.py`, `utils.py`, `pcqm4mv2_lmdb.py` (6-9-1) |
| 3단계 | 진행상황 출력 개선 (`tqdm`)    | `pcqm4mv2_lmdb.py` (3-5, 9-2, 10) |

자세한 구현 내용은 G~I 커밋을 참조하세요.
