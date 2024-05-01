# Model explanation with RISE implementation
This project implements RISE method to explain AI models.
Paper link to [RISE : Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/pdf/1806.07421)

We used two models trained on [Imagenette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz) a subset of ImageNet with 10 classes.
 * ResNet-34 pretrained on ImageNet-1k from [Microsoft](https://huggingface.co/microsoft/resnet-34)
 * MobileOne-s1 from [Apple mobile backbone](https://github.com/apple/ml-mobileone)


## Requirements :
- Have jupyter notebook on your computer

## User guide :
1. download or clone this respository
2. open `XAI_train.ipynb` : there is the training part of our two models
3. open `XAI_explanation.ipynb` : there is the implementation of explanation methods (**RISE**, **LIME**, **GradCAM**). We completely implemented RISE method following RISE paper instructions (see `rise.py`) and reused existing LIME and GradCAM functions.
4. open `XAI_evaluation.ipynb` : there is the evaluation of explanation methods using insertion and deletion metrics we implemented following RISE paper instructions (see `metrics.py`)
5. if you want to run the code, use the environment provided in `environment.yml`. To install this conda env, run `conda env create -f environment.yml`
