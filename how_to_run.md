### Dataset

- 전처리된 [PCQM4Mv2](https://huggingface.co/datasets/Syzseisus/cmc_dataset/tree/main/pcqm4mv2_lmdb_aug03_merged) 파일과 [MoleculeNet](https://huggingface.co/datasets/Syzseisus/cmc_dataset/blob/main/moleculenet.tar) 파일을 각각 다음 경로에 위치시켜주세요.
  - `pcqm4m-v2-train.sdf`는 [ogb](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)에서 받을 수 있습니다.
    ```
    wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz
    md5sum pcqm4m-v2-train.sdf.tar.gz # fd72bce606e7ddf36c2a832badeec6ab
    tar -xf pcqm4m-v2-train.sdf.tar.gz # extracted pcqm4m-v2-train.sdf
    ```
  - PCQM4Mv2 : `args.root` 폴더 안에 `pcqm4mv2_lmdb_aug03_merged` 위치
    - 다음과 같은 구조
      ```
      args.root/
      ├── pcqm4mv2_lmdb_aug03_merged
      │   ├── data.mdb
      │   └── lock.mdb
      └── pcqm4m-v2-train.sdf
      ```
  - MoleculeNet : `args.root` 폴더 안에 MoleculeNet의 각 데이터셋 폴더 위치
    - 각 데이터셋 폴더는 다음과 같은 구조
      ```
      args.root/{dataset_name}/
      ├── processed
      │   ├── data.pt
      │   ├── lmdb
      │   │   ├── data.mdb
      │   │   └── lock.mdb
      │   ├── pre_filter.pt
      │   └── pre_transform.pt
      └── raw
          └── bace.csv
      ```


### Pretrain

- `torchrun --nproc_per_node={NUM_GPUS} pretrain_lightning.py`
- pretrained ckpt는 [HuggingFace](https://huggingface.co/Syzseisus/cmc_ckpts/resolve/main/aug03_lambda_1.ckpt?download=true)에서 받을 수 있습니다.


### Finetuning

1. bash 파일 실행 권한 부여: `chmod +x ./run_all_moleculenet.sh`
2. bash 파일 실행 : `./run_all_moleculenet.sh {pretrain_ckpt_path}`
