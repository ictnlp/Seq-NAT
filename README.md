# Seq-NAT
Source code for &lt;Sequence-Level Training for Non-Autoregressive Neural Machine Translation>.
==================================
PyTorch implementation of the methods described in the paper [Sequence-Level Training for Non-Autoregressive Neural Machine Translation](). The code is based on fairseq v0.9.0. We only modified [nat_loss.py](https://github.com/ictnlp/Seq-NAT/blob/main/fairseq/criterions/nat_loss.py) and [utils.py](https://github.com/ictnlp/Seq-NAT/blob/main/fairseq/utils.py).

Dependencies
------------------
* Python 3.8
* PyTorch 1.7


Knowledge Distillation
------------------
Following [Gu et al. 2019](https://arxiv.org/abs/1905.11006), [knowledge distillation](https://arxiv.org/abs/1606.07947) from an autoregressive model can effectively simplify the training data distribution, which is sometimes essential for NAT-based models to learn good translations.
The easiest way of performing distillation is to follow the [instructions of training a standard transformer model](../translation) on the same data, and then decode the training set to produce a distillation dataset for NAT. Contact us if you need our distillation dataset.


Training
------------------
The training scripts are provided in the folder [training_scripts](https://github.com/ictnlp/Seq-NAT/tree/main/training_scripts). Firstly, run the pretraining script to pretrain the baseline NAT model:
```bash
$ sh training_scripts/pretrain.sh
```
Then, run other scripts for the finetuning. For example, to finetune the NAT model with the BoN-L1(N=2) objective, run:
```bash
$ sh training_scripts/bag2grams.sh
```
Decoding
------------------
To decode the test set, run:
```bash
$ sh decode.sh model_path
```


Citation
------------------
If you find the resources in this repository useful, please consider citing:
```

```
