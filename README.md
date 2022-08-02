# BRATS 2021 Solution For Segmentation Task
This repo contains the supported pytorch code and configuration files to reproduce 3D medical image segmentaion results of Reciprocal Adversarial Learning for Brain Tumor Segmentation: A Solution to BraTS Challenge 2021 Segmentation Task in [ArXiv](https://arxiv.org/pdf/2201.03777.pdf) and in [Springer Nature](https://link.springer.com/chapter/10.1007/978-3-031-08999-2_13). 


![Proposed Architecture](img/vizviva.png?raw=true)

## Environment
Prepare an environment with python=3.8, and then run the command "pip install -r requirements.txt" for the dependencies.

## Data Preparation
- File structure
    ```
     BRATS2021
      |---Data
      |   |--- RSNA_ASNR_MICCAI_BraTS2021_TrainingData
      |   |   |--- BraTS2021_00000
      |   |   |   |--- BraTS2021_00000_flair...
      |   
      |              
      |   
      |
      |---train.py
      |---test.py
      ...
    ```



## Train/Test
- Train : Run the train script on BraTS 2021 Training Dataset with Base model Configurations. 
```bash
python train.py --epochs 350
```

- Test : Run the test script on BraTS 2021 Training Dataset. 
```bash
python test.py
```
## Pre-trained Model
https://drive.google.com/file/d/11YmBPePPmnqE9W40ZqschovmiPx6lZ-2/view?usp=sharing

## Acknowledgements
This repository makes liberal use of code from [open_brats2020](https://github.com/lescientifik/open_brats2020).

## References
* [BraTS 2021](http://braintumorsegmentation.org/)

## Citing our work
```bash
   @misc{peiris2022reciprocal,
      title={Reciprocal Adversarial Learning for Brain Tumor Segmentation: A Solution to BraTS Challenge 2021 Segmentation Task}, 
      author={Himashi Peiris and Zhaolin Chen and Gary Egan and Mehrtash Harandi},
      year={2022},
      eprint={2201.03777},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
   }
    
```



