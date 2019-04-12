## Comparing the influence of Pre-trained word vectors on Text Classification Using CNN Through Keras on Tensorflow
This project compares the importance and influence of using pre-trained word vectors in the embedding layer. The model architecture implements the one proposed by Yoon Kim for sentence classification in his paper [**Convolutional Neural Networks for Sentence Classification**](https://arxiv.org/abs/1408.5882).

For comparing the performance, two models are designed and evaluated. The two models are exactly same with just one difference. In model 1, the embedding layer is **not** seeded with any pre-trained word vectors. Where as in model 2, the embedding layer is seeded with word vectors provided by [*GloVe*](http://nlp.stanford.edu/data/glove.6B.zip) and is made static during training.

The models are evaluated on test data set (TREC_10.label) provided by TREC. To make the comparison fair and to reduce the influence of randomness, the experiment is repeated five times (for each model) and the final accuracy is obtained as the average of the five observations.

| Model       | Accuracy   |
|-------------|------------|
|   Model 1 (**without** pre-trained vectors)  | 87.52 |
|   Model 2 (**with** pre-trained vectors)     | 89.56 |

## Observation:
Adding pre-trained word vectors clearly give an edge on the model performance. The accuracy is boosted by 2.04 by seeding the embedding layer with word vectors from GloVe. In other words, the model is able to generalize by exploiting the similarity of words which share close proximity in the word embedding space.


### Acknowledgments:
1. Yoon Kim - https://github.com/yoonkim/CNN_sentence
2. Denny Britz - https://github.com/dennybritz/cnn-text-classification-tf
3. Jason Brownlee - https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

### Foot Notes:
1. **GloVe embedding file is available [here](https://nlp.stanford.edu/data/glove.6B.zip) and should be kept inside `embeddings` folder.**
2. A validation split is not used and is intentional, since the aim here is to compare the performance on the models and not to identify the hyper parameters for the best performing model.
3. The accuracy from five runs are given below.

| Model                                        | Run1 | Run2 | Run3 | Run4 | Run5 | 
|----------------------------------------------|------|------|------|------|------|
|   Model 1 (**without** pre-trained vectors)  | 89.6 | 86.6 | 85.8 | 88   | 87.6 |
|   Model 2 (**with** pre-trained vectors)     | 90.4 | 90   | 89   | 88.6 | 89.8 |