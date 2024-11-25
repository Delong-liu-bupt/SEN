# SEN
> Text-guided Image Restoration and Semantic Enhancement for Text-to-Image Person Retrieval

### News
* [2023.08.01] Repo is created. Code will come soon.
* [2024.11.25] Code is now available! 🎉


```markdown
# SEN
> Text-guided Image Restoration and Semantic Enhancement for Text-to-Image Person Retrieval

### News
* [2023.08.01] Repo is created. Code will come soon.
* [2024.11.25] Code is now available! 🎉

## Usage

### Requirements
```
torch >= 1.10.0
torchvision >= 0.11.0
prettytable
easydict
nltk==3.8.1
```

### Prepare Datasets
Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), the ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN), and the RSTPReid dataset from [here](https://github.com/NjtechCVLab/RSTPReid-Dataset). Extract the downloaded datasets into the dataset root directory. The folder structure should be organized as follows:

```
|-- your data root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```

## Training

```bash
DATASET_NAME="CUHK-PEDES"
CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name sen \
--img_aug \
--batch_size 128 \
--need_MAE \
--mlm_loss_weight 1.0 \
--mae_loss_weight 10 \
--tri_loss_weight 10 \
--mask_ratio 0.7 \
--lr 1e-5 \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mae+id+tri' \
--root_dir 'your_data_path' \
--num_epoch 60 \
--lrscheduler 'cosine'
```

## Testing

```bash
python test.py --config_file 'path/to/model_dir/configs.yaml'
```

## SEN Results

### CUHK-PEDES dataset

|     Method      | Encoder              | Rank-1 | Rank-5 | Rank-10 |  mAP  |  mINP  |
|-----------------|----------------------|--------|--------|---------|-------|--------|
| CMPM/C          | RN50 + LSTM         | 49.37  | -      | 79.27   | -     | -      |
| PMA             | RN50 + LSTM         | 53.81  | 73.54  | 81.23   | -     | -      |
| TIMAM           | RN101 + BERT        | 54.51  | 77.56  | 79.27   | -     | -      |
| ViTAA           | RN50 + LSTM         | 54.92  | 75.18  | 82.90   | 51.60 | -      |
| NAFS            | RN50 + BERT         | 59.36  | 79.13  | 86.00   | 54.07 | -      |
| DSSL            | RN50 + BERT         | 59.98  | 80.41  | 87.56   | -     | -      |
| SSAN            | RN50 + LSTM         | 61.37  | 80.15  | 86.73   | -     | -      |
| LapsCore        | RN50 + BERT         | 63.40  | -      | 87.80   | -     | -      |
| ISANet          | RN50 + LSTM         | 63.92  | 82.15  | 87.69   | -     | -      |
| LBUL            | RN50 + BERT         | 64.04  | 82.66  | 87.22   | -     | -      |
| CM-MoCo         | CLIP-RN101 + CLIP-Xformer | 64.08  | 81.73  | 88.19   | 60.08 | -      |
| SAF             | ViT-Base + BERT     | 64.13  | 82.62  | 88.40   | -     | -      |
| TIPCB           | RN50 + BERT         | 64.26  | 83.19  | 89.10   | -     | -      |
| CAIBC           | RN50 + BERT         | 64.43  | 82.87  | 88.37   | -     | -      |
| AXM-Net         | RN50 + BERT         | 64.44  | 80.52  | 86.77   | 58.73 | -      |
| PBSL            | RN50 + BERT         | 65.32  | 83.81  | 89.26   | -     | -      |
| BEAT            | RN101 + BERT        | 65.61  | 83.45  | 89.57   | -     | -      |
| LGUR            | DeiT-Small + BERT   | 65.25  | 83.12  | 89.00   | -     | -      |
| IVT             | ViT-Base + BERT     | 65.59  | 83.11  | 89.21   | -     | -      |
| UniPT           | CLIP-ViT-B/16 + CLIP-Xformer | 68.50  | 84.67  | 90.38   | -     | -      |
| Gen             | ViT-Base + BERT     | 69.47  | 87.13  | 92.13   | 60.56 | -      |
| CFine           | CLIP-ViT-B/16 + BERT | 69.57 | 85.93  | 91.15   | -     | -      |
| IRRA            | CLIP-ViT-B/16 + CLIP-Xformer | 73.38  | 89.93  | 93.71   | 66.13 | 50.24  |
| **SEN (ours)**  | CLIP-ViT-B/16 + CLIP-Xformer | _75.00_ | _89.98_ | _94.09_ | _67.23_ | _51.45_ |
| **SEN-XL (ours)** | CLIP-ViT-L/14 + CLIP-Xformer | **76.64** | **91.33** | **94.66** | **69.19** | **53.88** |

### ICFG-PEDES dataset

|       Method       | Rank-1 | Rank-5 | Rank-10 |  mAP  |  mINP  |
|--------------------|--------|--------|---------|-------|--------|
| Dual Path          | 38.99  | 59.44  | 68.41   |   -   |   -    |
| CMPM/C             | 43.51  | 65.44  | 74.26   |   -   |   -    |
| ViTAA              | 50.98  | 68.79  | 75.78   |   -   |   -    |
| SSAN               | 54.23  | 72.63  | 79.53   |   -   |   -    |
| TIPCB              | 54.96  | 74.72  | 81.89   |   -   |   -    |
| IVT                | 56.04  | 73.60  | 80.22   |   -   |   -    |
| Gen                | 57.69  | 75.79  | 82.67   | 36.07 |   -    |
| ISANet             | 57.73  | 75.42  | 81.72   |   -   |   -    |
| PBSL               | 57.84  | 75.46  | 82.15   |   -   |   -    |
| BEAT               | 58.25  | 75.92  | 81.96   |   -   |   -    |
| UniPT              | 60.09  | 76.19  | 82.46   |   -   |   -    |
| CFine              | 60.83  | 76.55  | 82.42   |   -   |   -    |
| IRRA               | 63.46  | 80.25  | 85.82   | 38.06 |  7.93  |
| **SEN (ViT-B)**    | _64.56_ | _80.49_ | _85.88_ | _40.80_ | _9.51_ |
| **SEN-XL (ViT-L)** | **66.76** | **81.57** | **86.55** | **44.28** | **12.26** |

### RSTPReid dataset

|       Method       | Rank-1 | Rank-5 | Rank-10 |  mAP  |  mINP  |
|--------------------|--------|--------|---------|-------|--------|
| DSSL              | 39.05  | 62.60  | 73.95   |   -   |   -    |
| SSAN              | 43.50  | 67.80  | 77.15   |   -   |   -    |
| LBUL              | 45.55  | 68.20  | 77.85   |   -   |   -    |
| IVT               | 46.70  | 70.00  | 78.80   |   -   |   -    |
| PBSL              | 47.78  | 71.40  | 79.90   |   -   |   -    |
| BEAT              | 58.25  | 73.10  | 81.30   |   -   |   -    |
| CFine             | 50.55  | 72.50  | 81.60   |   -   |   -    |
| UniPT             | 51.85  | 74.85  | 82.85   |   -   |   -    |
| IRRA              | 60.20  | 81.30  | _88.20_ | 47.17 | 25.28  |
| **SEN (Ours)**    | _61.60_ | _81.75_ | 88.05   | _48.62_ | _26.43_ |
| **SEN-XL (ViT-L)** | **62.70** | **83.60** | **89.15** | **49.89** | **27.56** |

## Acknowledgments
Some components of this code implementation are adopted from [CLIP](https://github.com/openai/CLIP), [TransReID](https://github.com/damo-cv/TransReID), and [IRRA](https://github.com/anosorae/IRRA). We sincerely appreciate their contributions.

## Citation
If you find this code useful for your research, please cite our paper.

```tex
TBD
```
