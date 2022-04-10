# textContrastiveLib
A collection of text contrastive methods, reproducing existing text-based methods or application of CV domain methods.

## Model list
All encoders of these models are based on pretrained BERT.
 - SimCSE: [[original paper](https://arxiv.org/abs/2104.08821)] [[code](https://github.com/princeton-nlp/SimCSE)]
 <img src="/img/model-simcse.png" width="80%">

 - DirectCSE: [[original paper](https://arxiv.org/abs/2110.09348)]
 <img src="/img/model-directcse.png" width="50%">

 - BYOLSE: [[original paper](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)]
 <img src="/img/model-byolse.png" width="80%">

 - DirectBYOLSE: combination of DirectCSE and BYOLSE
 <img src="/img/model-directbyolse.png" width="50%">

## Requirement
- PyTorch 1.9
- python 3.8
- transformers 4.8
- jupyter-notebook

## Usage
1. prepare datasets (data-prepare notebook) and pretrained BERT weights
2. change all the file path in PATH.py to your own path
3. Follow the example-usage notebook to init&train models

Note:
only SimCSE model can get good results (STS ~ 76 ), other models are more of an experimental nature. 
Feel free to change the model architecture or training parameters for better results, try it!

## Bugs or questions?
If you have any questions or suggestions, feel free to open an issue!