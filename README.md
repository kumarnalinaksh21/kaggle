# TCS Enterprise Intelligent Automation – ARITIFICIAL INTELLIGENCE Competition
Repository for TCS Enterprise Intelligent Automation – ARITIFICIAL INTELLIGENCE Competition.

datasets are taken from Kaggle Quora competition. 

### Dependencies

* [gensim](https://radimrehurek.com/gensim/)
* [torch](http://pytorch.org/)


### Overview 
![Overview](https://github.com/kumarnalinaksh21/kaggle/blob/master/NetworkArchitecture.png "Solution Overview")

## Feature Extraction on Sentences
We extract doc2vec features on sentences using the *gensim* library and a pretrained doc2vec (DBOW) model trained on the English Wikipedia dataset.
The pretrained model is available [here](https://ibm.box.com/s/3f160t4xpuya9an935k84ig465gvymm2).

The code for feature extraction is here: [feat.py](https://github.com/kumarnalinaksh21/kaggle/blob/master/feat.py). 

To use it:
> python feat.py

## Network Training using doc2vec features with Softmax CrossEntropy Loss

The code is here: [train.py](https://github.com/kumarnalinaksh21/kaggle/blob/master/train.py).

To log per iteration loss use:
> bash train_with_logging.sh

Training logs are available in [log.info](https://github.com/kumarnalinaksh21/kaggle/blob/master/log.info).

![Training Loss](https://github.com/kumarnalinaksh21/kaggle/blob/master/lossplot.png "Training Loss")

## Testing the dataset using doc2vec features with trained network 

The code is here: [test.py](https://github.com/kumarnalinaksh21/kaggle/blob/master/test.py).

To use it:
> python test.py

The output (id_,prob_) is saved in the text file [test_probs.txt](https://github.com/kumarnalinaksh21/kaggle/blob/master/test_probs.txt).

### Low Level Details
* We eliminate all punctuation marks from the sentences in a preprocessing step
* We use input flipping dataset augmentation scheme: the network is robost to the order in which the two sentences are presented to it
* At test time we compute probabilities corresponding to the two input flips and these are averaged
* The network definition is available in [helpers/network.py](https://github.com/kumarnalinaksh21/kaggle/blob/master/helpers/network.py)
