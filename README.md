# Vizviva BRATS 2021 Solution
This repo contains the supported pytorch code and configuration files to reproduce 3D medical image segmentaion results of Reciprocal Adversarial Learning for Brain Tumor Segmentation: A Solution to BraTS Challenge 2021 Segmentation Task. 

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
python train.py --num_classes 3 --epochs 350
```

- Test : Run the test script on BraTS 2021 Training Dataset. 
```bash
python test.py --num_classes 3
```

## Acknowledgements
This repository makes liberal use of code from [open_brats2020](https://github.com/lescientifik/open_brats2020).

## References
* [BraTS 2021](http://braintumorsegmentation.org/)

## Citing VT-UNet
```bash
    
```



