# From ViT Features to Training-free Video Object Segmentation via Streaming-data Mixture Models
Official PyTorch implementation for our NeurIPS 2023 paper, [From ViT Features to Training-free Video Object Segmentation via Streaming-data Mixture Models](https://openreview.net/pdf?id=jfsjKBDB1z).

[Qualitative results](https://youtu.be/jZ6gtBIbzIc)

[Video presentation](https://youtu.be/sI12eC5D7qM)

![output3](https://github.com/BGU-CS-VIL/Training-Free-VOS/assets/23636745/f45e7fe2-27bd-420d-832d-b7995f0a8595)  ![output2](https://github.com/BGU-CS-VIL/Training-Free-VOS/assets/23636745/813dd250-b070-498d-9fa2-146b54302b34)
![output0](https://github.com/BGU-CS-VIL/Training-Free-VOS/assets/23636745/3a6cc58a-0c6a-41e9-bbe8-b49cb826d6e0)  ![output7](https://github.com/BGU-CS-VIL/Training-Free-VOS/assets/23636745/dc60244b-8b23-468a-93bd-952bbca5171d)
![output6](https://github.com/BGU-CS-VIL/Training-Free-VOS/assets/23636745/bd004216-4f79-418a-9de1-c3702f3cf472)  ![output5](https://github.com/BGU-CS-VIL/Training-Free-VOS/assets/23636745/038184fa-abdb-44ce-ab41-e1932b8a7a7b)



## Table of Contents
- [Installation](#installation)
- [Downloading DAVIS 2017 Dataset](#downloading-davis-2017-dataset)
- [Extracting Features with XCiT](#extracting-features-with-xcit)
- [Inference](#inference)
- [Citation](#citation)



## Installation

To create a repository using the `env.yaml` file, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/BGU-CS-VIL/Training-Free-VOS.git)https://github.com/BGU-CS-VIL/Training-Free-VOS.git

2. Navigate to the repository directory:
   ```bash
   cd Training-Free-VOS

4. Create an environment using the env.yaml file:
   ```bash
   conda env create -f env.yaml

5. Activate the environment:
   ```bash
   conda activate VOS
  
## Downloading DAVIS 2017 Dataset

Follow these steps to download and set up the DAVIS 2017 dataset:

1. Download the DAVIS 2017 dataset from the following link:
   [DAVIS 2017 TrainVal 480p](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip)

2. Extract the downloaded file under the `data` folder in your project directory:
   ```bash
   unzip DAVIS-2017-trainval-480p.zip -d ./data/

## Extracting Features with XCiT

We use the Cross-Covariance Image Transformer (XCiT) for feature extraction. You can find more information about XCiT here: [XCiT GitHub Repository](https://github.com/facebookresearch/xcit).

The pre-extracted features are available for download:

1. Download the features from the following link:
   [Download Features](https://drive.google.com/file/d/16cZdney_sErSZ5t8WDEI1RCUvGyM0mpK/view?usp=sharing)

2. Unzip the downloaded file into the `features` folder in your project directory:
   ```bash
   unzip XCIT-feat.zip -d ./features/

## Inference

To run the inference, use the command below. You can modify the arguments as needed:

```bash
python main_seg.py

--loc: Location scale factor. Default is 10.
--time: Time factor. Default is 0.33.
--num_models: Number of models. Default is 10.
--vis: Enable saving segmentation overlay on the image
```

## Citation
We hope you find our work useful. If you would like to acknowledge it in your project, please use the following citation:
```
@inproceedings{uziel2023vit,
  title={From ViT Features to Training-free Video Object Segmentation via Streaming-data Mixture Models},
  author={Uziel, Roy and Dinari, Or and Freifeld, Oren},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
