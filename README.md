# ABSA-PyTorch

> Aspect Based Sentiment Analysis, PyTorch Implementations.
>
> 基于方面的情感分析，使用PyTorch实现。

![Packagist](https://img.shields.io/packagist/l/doctrine/orm.svg) ![PRsWelcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

## Requirement

* PyTorch >= 0.4.0
* NumPy 1.13.3
* tensorboardX 1.2
* Python 3.6
* GloVe pre-trained word vectors (See `data_utils.py` for more detail)
  * Download pre-trained word vectors [here](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors),
  * extract the [glove.twitter.27B.zip](http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip) and [glove.42B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) to the root directory

## Usage

### Training

```sh
python train.py --model_name ian --dataset twitter --logdir ian_logs
```

#### See the training process (needs to install TensorFlow)

```sh
tensorboard --logdir=./ian_logs
```

### Inference

Please refer to [infer_example.py](./infer_example.py).

## Implemented models

### MGAN ([mgan.py](./models/mgan.py))
Fan, Feifan, et al. "Multi-grained Attention Network for Aspect-Level Sentiment Classification." Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2018. [[pdf]](http://aclweb.org/anthology/D18-1380)

![mgan](assets/mgan.png)

### AOA ([aoa.py](./models/aoa.py))
Huang, Binxuan, et al. "Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks." arXiv preprint arXiv:1804.06536 (2018). [[pdf]](https://arxiv.org/pdf/1804.06536.pdf)

![aoa](assets/aoa.png)

### TNet ([tnet_lf.py](./models/tnet_lf.py))
Li, Xin, et al. "Transformation Networks for Target-Oriented Sentiment Classification." arXiv preprint arXiv:1805.01086 (2018). [[pdf]](https://arxiv.org/pdf/1805.01086)

![tnet_lf](assets/tnet_lf.png)

### Cabasc ([cabasc.py](./models/cabasc.py))
Liu, Qiao, et al. "Content Attention Model for Aspect Based Sentiment Analysis." Proceedings of the 2018 World Wide Web Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2018.

![cabasc](assets/cabasc.png)


### RAM ([ram.py](./models/ram.py))
Chen, Peng, et al. "Recurrent Attention Network on Memory for Aspect Sentiment Analysis." Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. 2017. [[pdf]](http://www.aclweb.org/anthology/D17-1047)

![ram](assets/ram.png)


### MemNet ([memnet.py](./models/memnet.py))
Tang, Duyu, B. Qin, and T. Liu. "Aspect Level Sentiment Classification with Deep Memory Network." Conference on Empirical Methods in Natural Language Processing 2016:214-224. [[pdf]](https://arxiv.org/pdf/1605.08900)

![memnet](assets/memnet.png)


### IAN ([ian.py](./models/ian.py))
Ma, Dehong, et al. "Interactive Attention Networks for Aspect-Level Sentiment Classification." arXiv preprint arXiv:1709.00893 (2017). [[pdf]](https://arxiv.org/pdf/1709.00893)

![han](assets/han.png)

### ATAE-LSTM ([atae_lstm.py](./models/atae_lstm.py))
Wang, Yequan, Minlie Huang, and Li Zhao. "Attention-based lstm for aspect-level sentiment classification." Proceedings of the 2016 conference on empirical methods in natural language processing. 2016.

![han](assets/atae-lstm.png)


### TD-LSTM ([td_lstm.py](./models/td_lstm.py))

Tang, Duyu, et al. "Effective LSTMs for Target-Dependent Sentiment Classification." Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers. 2016. [[pdf]](https://arxiv.org/pdf/1512.01100)

![td-lstm](assets/td-lstm.png)


### LSTM ([lstm.py](./models/lstm.py))

![lstm](assets/lstm.png)


## Reviews / Surveys

Zhang, Lei, Shuai Wang, and Bing Liu. "Deep Learning for Sentiment Analysis: A Survey." arXiv preprint arXiv:1801.07883 (2018). [[pdf]](https://arxiv.org/pdf/1801.07883)

Young, Tom, et al. "Recent trends in deep learning based natural language processing." arXiv preprint arXiv:1708.02709 (2017). [[pdf]](https://arxiv.org/pdf/1708.02709)

## Contributions

Feel free to contribute!

You can raise an issue or submit a pull request, whichever is more convenient for you.

## WHAT I DO:
Try to use bert to help this model more robust.

Also change a lit bit to adopt to chinese version.
## Licence

MIT License
