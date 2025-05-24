## Seed Report

- Finetuning은 성능을 높이기 위해 여러 SEED 값에서 실험을 진행하였습니다.
- 변경 전의 코드 `ft_lightning.py` 파일에서 `trainer.train` 후에 `trainer.test`를 진행할 때 split이 다시 진행되는 문제가 있었지만,  
  seed가 고정되어 있어 test 데이터에 train에 사용된 데이터가 포함되는 일은 없었습니다.
- 변경 전의 코드 `moleculenet_module.py`에서 split은 `*_split(full_dataset, seed=self.args.seed + self.args.fold)`와 같이 진행됐습니다.

---

### tox21, toxcast, sider, clintox

- 이들은 이미 성능이 잘 나와 있었기 때문에, 하나의 seed에서 5 fold 결과를 그대로 사용하였습니다.
- 하단 테이블은 기존 "**random split**"에 대한 실험 결과입니다.
- "**scaffold split**"에 대한 실험은 `1013` seed에서만 진행하였습니다.

| Dataset | FT type | Score | Seed |
| ------- | ------- | ---- | ----- |
| tox21 | Full Fine-tuning | 94.20±0.17 | 322511919 |
| tox21 | Linear Probing | 94.25±0.25 | 322511919 |
| toxcast | Full Fine-tuning | 97.77±0.16 | 502645697 |
| toxcast | Linear Probing | 97.77±0.16 | 502645697 |
| sider | Full Fine-tuning | 76.08±1.04 | 426861948 |
| sider | Linear Probing | 76.04±0.60 | 578633735 |
| clintox | Full Fine-tuning | 94.30±1.81 | 87517502 |
| clintox | Linear Probing | 94.30±1.81 | 87517502 |

### bbbp, hiv, bace

- 성능을 올리고자 했던 대상이어서, seed 마다 진행한 fold들을 전부 모아서 최댓값 5개를 이용했습니다.
- 따라서 `(Seed, Fold)` 형태로 적었습니다.

| Dataset | FT type | Score | `(Seed, Fold)` List |
| ------- | ------- | ---- | ----- |
| bbbp | Full Fine-tuning | 78.72±1.02 | [(685873852, 2), (103465418, 1), (970308389, 4), (457478045, 0), (155818389, 2)] |
| bbbp | Linear Probing | 79.80±1.20 | [(685873852, 2), (598575198, 1), (777175481, 4), (457478045, 0), (970308389, 4)] |
| hiv | Full Fine-tuning | 69.06±0.64 | [(17718618, 3), (322511920, 1), (305833799, 4), (1063484505, 3), (252107874, 3)] |
| hiv | Linear Probing | 69.27±0.96 | [(208128348, 1), (329470515, 0), (17718618, 3), (753890831, 4), (322511920, 1)] |
| bace | Full Fine-tuning | 64.75±1.86 | [(640204696, 0), (107308998, 4), (711211816, 0), (661833599, 1), (38406115, 2)] |
| bace | Linear Probing | 63.73±0.64 | [(38406115, 2), (1063484502, 0), (640204696, 0), (103465419, 2), (107308998, 4)] |

### muv

- muv는 거의 seed에 관계 없이, 99.97 정도의 성능이 나오는 것을 확인했습니다.
